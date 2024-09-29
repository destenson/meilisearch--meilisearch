use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use rayon::iter::{MapInit, ParallelIterator};

pub trait ParallelIteratorExt: ParallelIterator {
    /// Maps items based on the init function.
    ///
    /// The init function is ran only as necessary which is basically once by thread.
    fn try_map_try_init<F, INIT, T, E, R>(
        self,
        init: INIT,
        map_op: F,
    ) -> MapInit<
        Self,
        impl Fn() -> Result<T, Arc<E>> + Sync + Send + Clone,
        impl Fn(&mut Result<T, Arc<E>>, Self::Item) -> Result<R, Arc<E>> + Sync + Send + Clone,
    >
    where
        E: Send + Sync,
        F: Fn(&mut T, Self::Item) -> Result<R, E> + Sync + Send + Clone,
        INIT: Fn() -> Result<T, E> + Sync + Send + Clone,
        R: Send,
    {
        self.map_init(
            move || match init() {
                Ok(t) => Ok(t),
                Err(err) => Err(Arc::new(err)),
            },
            move |maybe_t, item| match maybe_t {
                Ok(t) => map_op(t, item).map_err(Arc::new),
                Err(maybe_err) => Err(maybe_err.clone()),
            },
        )
    }
}

impl<T: ParallelIterator> ParallelIteratorExt for T {}

/// A pool of items that can be pull and generated on demand.
pub struct ItemsPool<F, T, E>
where
    F: Fn() -> Result<T, E>,
{
    init: F,
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<F, T, E> ItemsPool<F, T, E>
where
    F: Fn() -> Result<T, E>,
{
    /// Create a new unbounded items pool with the specified function
    /// to generate items when needed.
    ///
    /// The `init` function will be invoked whenever a call to `with` requires new items.
    pub fn new(init: F) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        ItemsPool { init, sender, receiver }
    }

    /// Consumes the pool to retrieve all remaining items.
    ///
    /// This method is useful for cleaning up and managing the items once they are no longer needed.
    pub fn into_items(self) -> crossbeam_channel::IntoIter<T> {
        self.receiver.into_iter()
    }

    /// Allows running a function on an item from the pool,
    /// potentially generating a new item if the pool is empty.
    pub fn with<G, R>(&self, f: G) -> Result<R, E>
    where
        G: FnOnce(&mut T) -> Result<R, E>,
    {
        let mut item = match self.receiver.try_recv() {
            Ok(item) => item,
            Err(TryRecvError::Empty) => (self.init)()?,
            Err(TryRecvError::Disconnected) => unreachable!(),
        };

        // Run the user's closure with the retrieved item
        let result = f(&mut item);

        if let Err(e) = self.sender.send(item) {
            unreachable!("error when sending into channel {e}");
        }

        result
    }
}
