use std::sync::Arc;

use heed::RoTxn;
use rayon::iter::{IndexedParallelIterator, ParallelBridge, ParallelIterator};
use rhai::{Dynamic, Engine, OptimizationLevel, Scope};
use roaring::RoaringBitmap;
use tracing::field;

use super::DocumentChanges;
use crate::error::{FieldIdMapMissingEntry, InternalError};
use crate::update::new::{Deletion, DocumentChange, KvReaderFieldId, Update};
use crate::{
    all_obkv_to_json, Error, FieldsIdsMap, Index, InternalError, Object, Result, UserError,
};

pub struct UpdateByFunction {
    documents: RoaringBitmap,
    context: Option<Object>,
    code: String,
}

impl UpdateByFunction {
    pub fn new(documents: RoaringBitmap, context: Option<Object>, code: String) -> Self {
        UpdateByFunction { documents, context, code }
    }
}

impl<'p> DocumentChanges<'p> for UpdateByFunction {
    type Parameter = (&'p Index, &'p RoTxn<'p>, &'p FieldsIdsMap);

    fn document_changes(
        self,
        _fields_ids_map: &mut FieldsIdsMap,
        param: Self::Parameter,
    ) -> Result<
        impl IndexedParallelIterator<Item = std::result::Result<DocumentChange, Arc<Error>>>
            + Clone
            + 'p,
    > {
        let (index, rtxn, fields_ids_map) = param;

        // Setup the security and limits of the Engine
        let mut engine = Engine::new();
        engine.set_optimization_level(OptimizationLevel::Full);
        engine.set_max_call_levels(1000);
        // It is an arbitrary value. We need to let users define this in the settings.
        engine.set_max_operations(1_000_000);
        engine.set_max_variables(1000);
        engine.set_max_functions(30);
        engine.set_max_expr_depths(100, 1000);
        engine.set_max_string_size(1024 * 1024 * 1024); // 1 GiB
        engine.set_max_array_size(10_000);
        engine.set_max_map_size(10_000);

        let ast = engine.compile(self.code).map_err(UserError::DocumentEditionCompilationError)?;
        let context: Option<Dynamic> = match self.context {
            Some(context) => {
                Some(serde_json::from_value(context.into()).map_err(InternalError::SerdeJson)?)
            }
            None => None,
        };

        Ok(self.documents.into_iter().par_bridge().map(move |docid| {
            // safety: Both documents *must* exists in the database as
            //         their IDs comes from the list of documents ids.
            let current_document = index.document(rtxn, docid)?;
            let rhai_document = obkv_to_rhaimap(current_document, fields_ids_map)?;
            let json_document = all_obkv_to_json(current_document, fields_ids_map)?;
            let document_id = &json_document[primary_key];

            let mut scope = Scope::new();
            if let Some(context) = context.as_ref().cloned() {
                scope.push_constant_dynamic("context", context.clone());
            }
            scope.push("doc", rhai_document);
            // That's were the magic happens. We run the user script
            // which edits "doc" scope variable reprensenting the document
            // and ignore the output and even the type of it, i.e., Dynamic.
            let _ = engine
                .eval_ast_with_scope::<Dynamic>(&mut scope, &ast)
                .map_err(UserError::DocumentEditionRuntimeError)?;

            match scope.remove::<Dynamic>("doc") {
                // If the "doc" variable has set to (), we effectively delete the document.
                Some(doc) if doc.is_unit() => Ok(DocumentChange::Deletion(Deletion::create(
                    docid,
                    document_id,
                    current_document.boxed(),
                ))),
                None => unreachable!("missing doc variable from the Rhai scope"),
                Some(new_document) => match new_document.try_cast() {
                    Some(new_document) => {
                        let new_document = rhaimap_to_object(new_document);
                        // Note: This condition is not perfect. Sometimes it detect changes
                        //       like with floating points numbers and consider updating
                        //       the document even if nothing actually changed.
                        if json_document != new_document {
                            if Some(document_id) != new_document.get(primary_key) {
                                Err(Error::UserError(
                                    UserError::DocumentEditionCannotModifyPrimaryKey,
                                ))
                            } else {
                                let new = todo!();
                                Ok(DocumentChange::Update(Update::create(
                                    docid,
                                    document_id,
                                    current_document.boxed(),
                                    new,
                                )))
                            }
                        } else {
                            // TODO be smarter about this, nothing change ignore it.
                            Ok(DocumentChange::Update(Update::create(
                                docid,
                                document_id,
                                current_document.boxed(),
                                current_document.boxed(),
                            )))
                        }
                    }
                    None => Err(Error::UserError(UserError::DocumentEditionDocumentMustBeObject)),
                },
            }
        }))
    }
}

fn obkv_to_rhaimap(obkv: &KvReaderFieldId, fields_ids_map: &FieldsIdsMap) -> Result<rhai::Map> {
    let all_keys = obkv.iter().map(|(k, _v)| k).collect::<Vec<_>>();
    let map: Result<rhai::Map> = all_keys
        .iter()
        .copied()
        .flat_map(|id| obkv.get(id).map(|value| (id, value)))
        .map(|(id, value)| {
            let name = fields_ids_map.name(id).ok_or(FieldIdMapMissingEntry::FieldId {
                field_id: id,
                process: "all_obkv_to_rhaimap",
            })?;
            let value = serde_json::from_slice(value).map_err(InternalError::SerdeJson)?;
            Ok((name.into(), value))
        })
        .collect();

    map
}

fn rhaimap_to_object(map: rhai::Map) -> Object {
    let mut output = Object::new();
    for (key, value) in map {
        let value = serde_json::to_value(&value).unwrap();
        output.insert(key.into(), value);
    }
    output
}
