use std::sync::{Arc, RwLock};
use std::thread::{self, Builder};

use big_s::S;
pub use document_deletion::DocumentDeletion;
pub use document_operation::DocumentOperation;
use heed::{RoTxn, RwTxn};
pub use partial_dump::PartialDump;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::ThreadPool;
use time::OffsetDateTime;
pub use update_by_function::UpdateByFunction;

use super::channel::*;
use super::document_change::{Deletion, DocumentChange, Insertion, Update};
use super::extract::*;
use super::merger::{merge_grenad_entries, FacetFieldIdsDelta};
use super::word_fst_builder::PrefixDelta;
use super::words_prefix_docids::{
    compute_word_prefix_docids, compute_word_prefix_fid_docids, compute_word_prefix_position_docids,
};
use super::{StdResult, TopLevelMap};
use crate::documents::{PrimaryKey, DEFAULT_PRIMARY_KEY};
use crate::facet::FacetType;
use crate::proximity::ProximityPrecision;
use crate::update::new::channel::ExtractorSender;
use crate::update::new::parallel_iterator_ext::ParallelIteratorExt;
use crate::update::new::words_prefix_docids::compute_exact_word_prefix_docids;
use crate::update::settings::InnerIndexSettings;
use crate::update::{FacetsUpdateBulk, GrenadParameters};
use crate::{Error, FieldsIdsMap, GlobalFieldsIdsMap, Index, Result, UserError};

mod document_deletion;
mod document_operation;
mod partial_dump;
mod update_by_function;

pub trait DocumentChanges<'p> {
    type Parameter: 'p;

    fn document_changes(
        self,
        fields_ids_map: &mut FieldsIdsMap,
        param: Self::Parameter,
    ) -> Result<
        impl IndexedParallelIterator<Item = std::result::Result<DocumentChange, Arc<Error>>>
            + Clone
            + 'p,
    >;
}

/// This is the main function of this crate.
///
/// Give it the output of the [`Indexer::document_changes`] method and it will execute it in the [`rayon::ThreadPool`].
///
/// TODO return stats
pub fn index<PI>(
    wtxn: &mut RwTxn,
    index: &Index,
    fields_ids_map: FieldsIdsMap,
    pool: &ThreadPool,
    document_changes: PI,
) -> Result<()>
where
    PI: IndexedParallelIterator<Item = std::result::Result<DocumentChange, Arc<Error>>>
        + Send
        + Clone,
{
    let (merger_sender, writer_receiver) = merger_writer_channel(10_000);
    // This channel acts as a rendezvous point to ensure that we are one task ahead
    let (extractor_sender, merger_receiver) = extractors_merger_channels(4);

    let fields_ids_map_lock = RwLock::new(fields_ids_map);
    let global_fields_ids_map = GlobalFieldsIdsMap::new(&fields_ids_map_lock);
    let global_fields_ids_map_clone = global_fields_ids_map.clone();

    thread::scope(|s| {
        let indexer_span = tracing::Span::current();
        // TODO manage the errors correctly
        let handle = Builder::new().name(S("indexer-extractors")).spawn_scoped(s, move || {
            pool.in_place_scope(|_s| {
                    let span = tracing::trace_span!(target: "indexing::documents", parent: &indexer_span, "extract");
                    let _entered = span.enter();
                    let document_changes = document_changes.into_par_iter();

                    // document but we need to create a function that collects and compresses documents.
                    let document_sender = extractor_sender.document_sender();
                    document_changes.clone().into_par_iter().try_arc_for_each::<_, Error>(
                        |result| {
                        match result? {
                            DocumentChange::Deletion(Deletion { docid, external_document_id, ..}) => {
                                document_sender.delete(docid, external_document_id).unwrap();
                            }
                            DocumentChange::Update(Update { docid, external_document_id, new, ..}) => {
                                document_sender.insert(docid, external_document_id, new).unwrap();
                            }
                            DocumentChange::Insertion(Insertion { docid, external_document_id, new, ..}) => {
                                document_sender.insert(docid, external_document_id, new).unwrap();
                                // extracted_dictionary_sender.send(self, dictionary: &[u8]);
                            }
                        }
                        Ok(())
                    })?;

                    document_sender.finish().unwrap();

                    const TEN_GIB: usize = 10 * 1024 * 1024 * 1024;
                    let max_memory = TEN_GIB / dbg!(rayon::current_num_threads());
                    let grenad_parameters = GrenadParameters {
                        max_memory: Some(max_memory),
                        ..GrenadParameters::default()
                    };

                    {
                        let span = tracing::trace_span!(target: "indexing::documents::extract", "faceted");
                        let _entered = span.enter();
                        extract_and_send_docids::<
                            FacetedDocidsExtractor,
                            FacetDocids,
                        >(
                            index,
                            &global_fields_ids_map,
                            grenad_parameters,
                            document_changes.clone(),
                            &extractor_sender,
                        )?;
                    }

                    {
                        let span = tracing::trace_span!(target: "indexing::documents::extract", "word_docids");
                        let _entered = span.enter();

                        let WordDocidsMergers {
                            word_fid_docids,
                            word_docids,
                            exact_word_docids,
                            word_position_docids,
                            fid_word_count_docids,
                        } = WordDocidsExtractors::run_extraction(index, &global_fields_ids_map, grenad_parameters, document_changes.clone())?;
                        extractor_sender.send_searchable::<WordDocids>(word_docids).unwrap();
                        extractor_sender.send_searchable::<WordFidDocids>(word_fid_docids).unwrap();
                        extractor_sender.send_searchable::<ExactWordDocids>(exact_word_docids).unwrap();
                        extractor_sender.send_searchable::<WordPositionDocids>(word_position_docids).unwrap();
                        extractor_sender.send_searchable::<FidWordCountDocids>(fid_word_count_docids).unwrap();
                    }

                    // run the proximity extraction only if the precision is by word
                    // this works only if the settings didn't change during this transaction.
                    let rtxn = index.read_txn().unwrap();
                    let proximity_precision = index.proximity_precision(&rtxn)?.unwrap_or_default();
                    if proximity_precision == ProximityPrecision::ByWord {
                        let span = tracing::trace_span!(target: "indexing::documents::extract", "word_pair_proximity_docids");
                        let _entered = span.enter();
                        extract_and_send_docids::<
                            WordPairProximityDocidsExtractor,
                            WordPairProximityDocids,
                        >(
                            index,
                            &global_fields_ids_map,
                            grenad_parameters,
                            document_changes.clone(),
                            &extractor_sender,
                        )?;
                    }

                    {
                        let span = tracing::trace_span!(target: "indexing::documents::extract", "FINISH");
                        let _entered = span.enter();
                    }

                    // TODO THIS IS TOO MUCH
                    // - [ ] Extract fieldid docid facet number
                    // - [ ] Extract fieldid docid facet string
                    // - [ ] Extract facetid string fst
                    // - [ ] Extract facetid normalized string strings

                    // TODO Inverted Indexes again
                    // - [x] Extract fieldid facet isempty docids
                    // - [x] Extract fieldid facet isnull docids
                    // - [x] Extract fieldid facet exists docids

                    // TODO This is the normal system
                    // - [x] Extract fieldid facet number docids
                    // - [x] Extract fieldid facet string docids

                    Ok(()) as Result<_>
                })
        })?;

        let indexer_span = tracing::Span::current();
        // TODO manage the errors correctly
        let merger_thread = Builder::new().name(S("indexer-merger")).spawn_scoped(s, move || {
            let span =
                tracing::trace_span!(target: "indexing::documents", parent: &indexer_span, "merge");
            let _entered = span.enter();
            let rtxn = index.read_txn().unwrap();
            merge_grenad_entries(
                merger_receiver,
                merger_sender,
                &rtxn,
                index,
                global_fields_ids_map_clone,
            )
        })?;

        for operation in writer_receiver {
            let database = operation.database(index);
            match operation.entry() {
                EntryOperation::Delete(e) => {
                    if !database.delete(wtxn, e.entry())? {
                        unreachable!("We tried to delete an unknown key")
                    }
                }
                EntryOperation::Write(e) => database.put(wtxn, e.key(), e.value())?,
            }
        }

        /// TODO handle the panicking threads
        handle.join().unwrap()?;
        let merger_result = merger_thread.join().unwrap()?;

        if let Some(facet_field_ids_delta) = merger_result.facet_field_ids_delta {
            compute_facet_level_database(index, wtxn, facet_field_ids_delta)?;
        }

        if let Some(prefix_delta) = merger_result.prefix_delta {
            compute_prefix_database(index, wtxn, prefix_delta)?;
        }

        Ok(()) as Result<_>
    })?;

    let fields_ids_map = fields_ids_map_lock.into_inner().unwrap();
    index.put_fields_ids_map(wtxn, &fields_ids_map)?;

    // used to update the localized and weighted maps while sharing the update code with the settings pipeline.
    let mut inner_index_settings = InnerIndexSettings::from_index(index, wtxn)?;
    inner_index_settings.recompute_facets(wtxn, index)?;
    inner_index_settings.recompute_searchables(wtxn, index)?;

    index.set_updated_at(wtxn, &OffsetDateTime::now_utc())?;

    Ok(())
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing::prefix")]
fn compute_prefix_database(
    index: &Index,
    wtxn: &mut RwTxn,
    prefix_delta: PrefixDelta,
) -> Result<()> {
    eprintln!("prefix_delta: {:?}", &prefix_delta);
    let PrefixDelta { modified, deleted } = prefix_delta;
    // Compute word prefix docids
    compute_word_prefix_docids(wtxn, index, &modified, &deleted)?;
    // Compute exact word prefix docids
    compute_exact_word_prefix_docids(wtxn, index, &modified, &deleted)?;
    // Compute word prefix fid docids
    compute_word_prefix_fid_docids(wtxn, index, &modified, &deleted)?;
    // Compute word prefix position docids
    compute_word_prefix_position_docids(wtxn, index, &modified, &deleted)
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing::facet_field_ids")]
fn compute_facet_level_database(
    index: &Index,
    wtxn: &mut RwTxn,
    facet_field_ids_delta: FacetFieldIdsDelta,
) -> Result<()> {
    eprintln!("facet_field_ids_delta: {:?}", &facet_field_ids_delta);
    if let Some(modified_facet_string_ids) = facet_field_ids_delta.modified_facet_string_ids() {
        let span = tracing::trace_span!(target: "indexing::facet_field_ids", "string");
        let _entered = span.enter();
        FacetsUpdateBulk::new_not_updating_level_0(
            index,
            modified_facet_string_ids,
            FacetType::String,
        )
        .execute(wtxn)?;
    }
    if let Some(modified_facet_number_ids) = facet_field_ids_delta.modified_facet_number_ids() {
        let span = tracing::trace_span!(target: "indexing::facet_field_ids", "number");
        let _entered = span.enter();
        FacetsUpdateBulk::new_not_updating_level_0(
            index,
            modified_facet_number_ids,
            FacetType::Number,
        )
        .execute(wtxn)?;
    }

    Ok(())
}

/// TODO: GrenadParameters::default() should be removed in favor a passed parameter
/// TODO: manage the errors correctly
/// TODO: we must have a single trait that also gives the extractor type
fn extract_and_send_docids<E: DocidsExtractor, D: MergerOperationType>(
    index: &Index,
    fields_ids_map: &GlobalFieldsIdsMap,
    indexer: GrenadParameters,
    document_changes: impl IntoParallelIterator<Item = std::result::Result<DocumentChange, Arc<Error>>>,
    sender: &ExtractorSender,
) -> Result<()> {
    let merger = E::run_extraction(index, fields_ids_map, indexer, document_changes)?;
    sender.send_searchable::<D>(merger).unwrap();
    Ok(())
}

/// Returns the primary key *field id* that has already been set for this index or the
/// one we will guess by searching for the first key that contains "id" as a substring.
/// TODO move this elsewhere
pub fn retrieve_or_guess_primary_key<'a>(
    rtxn: &'a RoTxn<'a>,
    index: &Index,
    fields_ids_map: &mut FieldsIdsMap,
    first_document: Option<&'a TopLevelMap<'_>>,
) -> Result<StdResult<PrimaryKey<'a>, UserError>> {
    match index.primary_key(rtxn)? {
        Some(primary_key) => match PrimaryKey::new(primary_key, fields_ids_map) {
            Some(primary_key) => Ok(Ok(primary_key)),
            None => unreachable!("Why is the primary key not in the fidmap?"),
        },
        None => {
            let first_document = match first_document {
                Some(document) => document,
                None => return Ok(Err(UserError::NoPrimaryKeyCandidateFound)),
            };

            let mut guesses: Vec<&str> = first_document
                .keys()
                .map(AsRef::as_ref)
                .filter(|name| name.to_lowercase().ends_with(DEFAULT_PRIMARY_KEY))
                .collect();

            // sort the keys in lexicographical order, so that fields are always in the same order.
            guesses.sort_unstable();

            match guesses.as_slice() {
                [] => Ok(Err(UserError::NoPrimaryKeyCandidateFound)),
                [name] => {
                    tracing::info!("Primary key was not specified in index. Inferred to '{name}'");
                    match fields_ids_map.insert(name) {
                        Some(field_id) => Ok(Ok(PrimaryKey::Flat { name, field_id })),
                        None => Ok(Err(UserError::AttributeLimitReached)),
                    }
                }
                multiple => Ok(Err(UserError::MultiplePrimaryKeyCandidatesFound {
                    candidates: multiple.iter().map(|candidate| candidate.to_string()).collect(),
                })),
            }
        }
    }
}
