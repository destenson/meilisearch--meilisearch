use std::cmp::Ordering;
use std::sync::{OnceLock, RwLock};
use std::thread::{self, Builder};

use big_s::S;
use document_changes::{extract, DocumentChanges, IndexingContext, Progress};
pub use document_deletion::DocumentDeletion;
pub use document_operation::{DocumentOperation, PayloadStats};
use hashbrown::HashMap;
use heed::types::{Bytes, DecodeIgnore, Str};
use heed::{RoTxn, RwTxn};
use itertools::{merge_join_by, EitherOrBoth};
pub use partial_dump::PartialDump;
use rand::SeedableRng as _;
use raw_collections::RawMap;
use time::OffsetDateTime;
pub use update_by_function::UpdateByFunction;

use super::channel::*;
use super::extract::*;
use super::facet_search_builder::FacetSearchBuilder;
use super::merger::FacetFieldIdsDelta;
use super::steps::Step;
use super::thread_local::ThreadLocal;
use super::word_fst_builder::{PrefixData, PrefixDelta, WordFstBuilder};
use super::words_prefix_docids::{
    compute_word_prefix_docids, compute_word_prefix_fid_docids, compute_word_prefix_position_docids,
};
use super::StdResult;
use crate::documents::{PrimaryKey, DEFAULT_PRIMARY_KEY};
use crate::facet::FacetType;
use crate::fields_ids_map::metadata::{FieldIdMapWithMetadata, MetadataBuilder};
use crate::index::main_key::{WORDS_FST_KEY, WORDS_PREFIXES_FST_KEY};
use crate::proximity::ProximityPrecision;
use crate::update::del_add::DelAdd;
use crate::update::new::extract::EmbeddingExtractor;
use crate::update::new::merger::merge_and_send_rtree;
use crate::update::new::words_prefix_docids::compute_exact_word_prefix_docids;
use crate::update::new::{merge_and_send_docids, merge_and_send_facet_docids, FacetDatabases};
use crate::update::settings::InnerIndexSettings;
use crate::update::{FacetsUpdateBulk, GrenadParameters};
use crate::vector::{ArroyWrapper, EmbeddingConfigs, Embeddings};
use crate::{
    Error, FieldsIdsMap, GlobalFieldsIdsMap, Index, InternalError, Result, ThreadPoolNoAbort,
    ThreadPoolNoAbortBuilder, UserError,
};

pub(crate) mod de;
pub mod document_changes;
mod document_deletion;
mod document_operation;
mod partial_dump;
mod update_by_function;

/// This is the main function of this crate.
///
/// Give it the output of the [`Indexer::document_changes`] method and it will execute it in the [`rayon::ThreadPool`].
///
/// TODO return stats
#[allow(clippy::too_many_arguments)] // clippy: 😝
pub fn index<'pl, 'indexer, 'index, DC, MSP, SP>(
    wtxn: &mut RwTxn,
    index: &'index Index,
    grenad_parameters: GrenadParameters,
    db_fields_ids_map: &'indexer FieldsIdsMap,
    new_fields_ids_map: FieldsIdsMap,
    new_primary_key: Option<PrimaryKey<'pl>>,
    document_changes: &DC,
    embedders: EmbeddingConfigs,
    must_stop_processing: &'indexer MSP,
    send_progress: &'indexer SP,
) -> Result<()>
where
    DC: DocumentChanges<'pl>,
    MSP: Fn() -> bool + Sync,
    SP: Fn(Progress) + Sync,
{
    let (extractor_sender, writer_receiver) = extractor_writer_channel(10_000);

    let metadata_builder = MetadataBuilder::from_index(index, wtxn)?;
    let new_fields_ids_map = FieldIdMapWithMetadata::new(new_fields_ids_map, metadata_builder);
    let new_fields_ids_map = RwLock::new(new_fields_ids_map);
    let fields_ids_map_store = ThreadLocal::with_capacity(rayon::current_num_threads());
    let mut extractor_allocs = ThreadLocal::with_capacity(rayon::current_num_threads());
    let doc_allocs = ThreadLocal::with_capacity(rayon::current_num_threads());

    let indexing_context = IndexingContext {
        index,
        db_fields_ids_map,
        new_fields_ids_map: &new_fields_ids_map,
        doc_allocs: &doc_allocs,
        fields_ids_map_store: &fields_ids_map_store,
        must_stop_processing,
        send_progress,
    };

    let mut field_distribution = index.field_distribution(wtxn)?;
    let mut document_ids = index.documents_ids(wtxn)?;

    thread::scope(|s| -> Result<()> {
        let indexer_span = tracing::Span::current();
        let embedders = &embedders;
        // prevent moving the field_distribution and document_ids in the inner closure...
        let field_distribution = &mut field_distribution;
        let document_ids = &mut document_ids;
        let extractor_handle = Builder::new().name(S("indexer-extractors")).spawn_scoped(s, move || {
            let span = tracing::trace_span!(target: "indexing::documents", parent: &indexer_span, "extract");
            let _entered = span.enter();

            let rtxn = index.read_txn()?;


            // document but we need to create a function that collects and compresses documents.
            let document_sender = extractor_sender.documents();
            let document_extractor = DocumentsExtractor::new(&document_sender, embedders);
            let datastore = ThreadLocal::with_capacity(rayon::current_num_threads());
            {
                let span = tracing::trace_span!(target: "indexing::documents::extract", parent: &indexer_span, "documents");
                let _entered = span.enter();
                extract(document_changes,
                    &document_extractor,
                    indexing_context,
                    &mut extractor_allocs,
                    &datastore,
                    Step::ExtractingDocuments,
                )?;
            }
            {
                let span = tracing::trace_span!(target: "indexing::documents::merge", parent: &indexer_span, "documents");
                let _entered = span.enter();
                for document_extractor_data in datastore {
                    let document_extractor_data = document_extractor_data.0.into_inner();
                    for (field, delta) in document_extractor_data.field_distribution_delta {
                        let current = field_distribution.entry(field).or_default();
                        // adding the delta should never cause a negative result, as we are removing fields that previously existed.
                        *current = current.saturating_add_signed(delta);
                    }
                    document_extractor_data.docids_delta.apply_to(document_ids);
                }

                field_distribution.retain(|_, v| *v != 0);
            }

            let facet_field_ids_delta;

            {
                let caches = {
                    let span = tracing::trace_span!(target: "indexing::documents::extract", parent: &indexer_span, "faceted");
                    let _entered = span.enter();

                    FacetedDocidsExtractor::run_extraction(
                            grenad_parameters,
                            document_changes,
                            indexing_context,
                            &mut extractor_allocs,
                            &extractor_sender.field_id_docid_facet_sender(),
                            Step::ExtractingFacets
                        )?
                };

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", parent: &indexer_span, "faceted");
                    let _entered = span.enter();

                    facet_field_ids_delta = merge_and_send_facet_docids(
                        caches,
                        FacetDatabases::new(index),
                        index,
                        extractor_sender.facet_docids(),
                    )?;
                }
            }

            {




                let WordDocidsCaches {
                    word_docids,
                    word_fid_docids,
                    exact_word_docids,
                    word_position_docids,
                    fid_word_count_docids,
                } = {
                    let span = tracing::trace_span!(target: "indexing::documents::extract", "word_docids");
                    let _entered = span.enter();

                    WordDocidsExtractors::run_extraction(
                        grenad_parameters,
                        document_changes,
                        indexing_context,
                        &mut extractor_allocs,
                        Step::ExtractingWords
                    )?
                };

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "word_docids");
                    let _entered = span.enter();
                    merge_and_send_docids(
                        word_docids,
                        index.word_docids.remap_types(),
                        index,
                        extractor_sender.docids::<WordDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "word_fid_docids");
                    let _entered = span.enter();
                    merge_and_send_docids(
                        word_fid_docids,
                        index.word_fid_docids.remap_types(),
                        index,
                        extractor_sender.docids::<WordFidDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "exact_word_docids");
                    let _entered = span.enter();
                    merge_and_send_docids(
                        exact_word_docids,
                        index.exact_word_docids.remap_types(),
                        index,
                        extractor_sender.docids::<ExactWordDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "word_position_docids");
                    let _entered = span.enter();
                    merge_and_send_docids(
                        word_position_docids,
                        index.word_position_docids.remap_types(),
                        index,
                        extractor_sender.docids::<WordPositionDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "fid_word_count_docids");
                    let _entered = span.enter();
                    merge_and_send_docids(
                        fid_word_count_docids,
                        index.field_id_word_count_docids.remap_types(),
                        index,
                        extractor_sender.docids::<FidWordCountDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }
            }

            // run the proximity extraction only if the precision is by word
            // this works only if the settings didn't change during this transaction.
            let proximity_precision = index.proximity_precision(&rtxn)?.unwrap_or_default();
            if proximity_precision == ProximityPrecision::ByWord {
                let caches = {
                    let span = tracing::trace_span!(target: "indexing::documents::extract", "word_pair_proximity_docids");
                    let _entered = span.enter();

                    <WordPairProximityDocidsExtractor as DocidsExtractor>::run_extraction(
                        grenad_parameters,
                        document_changes,
                        indexing_context,
                        &mut extractor_allocs,
                        Step::ExtractingWordProximity,
                    )?
                };

                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "word_pair_proximity_docids");
                    let _entered = span.enter();

                    merge_and_send_docids(
                        caches,
                        index.word_pair_proximity_docids.remap_types(),
                        index,
                        extractor_sender.docids::<WordPairProximityDocids>(),
                        &indexing_context.must_stop_processing,
                    )?;
                }
            }

            'vectors: {

                let mut index_embeddings = index.embedding_configs(&rtxn)?;
                if index_embeddings.is_empty() {
                    break 'vectors;
                }

                let embedding_sender = extractor_sender.embeddings();
                let extractor = EmbeddingExtractor::new(embedders, &embedding_sender, field_distribution, request_threads());
                let mut datastore = ThreadLocal::with_capacity(rayon::current_num_threads());
                {
                    let span = tracing::trace_span!(target: "indexing::documents::extract", "vectors");
                    let _entered = span.enter();

                    extract(document_changes, &extractor, indexing_context, &mut extractor_allocs, &datastore, Step::ExtractingEmbeddings)?;
                }
                {
                    let span = tracing::trace_span!(target: "indexing::documents::merge", "vectors");
                    let _entered = span.enter();

                    for config in &mut index_embeddings {
                        'data: for data in datastore.iter_mut() {
                            let data = &mut data.get_mut().0;
                            let Some(deladd) = data.remove(&config.name) else { continue 'data; };
                            deladd.apply_to(&mut config.user_provided);
                        }
                    }
                }

                embedding_sender.finish(index_embeddings).unwrap();
            }

            'geo: {
                let Some(extractor) = GeoExtractor::new(&rtxn, index, grenad_parameters)? else {
                    break 'geo;
                };
                let datastore = ThreadLocal::with_capacity(rayon::current_num_threads());

                {
                    let span = tracing::trace_span!(target: "indexing::documents::extract", "geo");
                    let _entered = span.enter();

                    extract(
                        document_changes,
                        &extractor,
                        indexing_context,
                        &mut extractor_allocs,
                        &datastore,
                        Step::WritingGeoPoints
                    )?;
                }

                merge_and_send_rtree(
                    datastore,
                    &rtxn,
                    index,
                    extractor_sender.geo(),
                    &indexing_context.must_stop_processing,
                )?;
            }

            (indexing_context.send_progress)(Progress::from_step(Step::WritingToDatabase));

            Result::Ok(facet_field_ids_delta)
        })?;

        let global_fields_ids_map = GlobalFieldsIdsMap::new(&new_fields_ids_map);

        let vector_arroy = index.vector_arroy;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let indexer_span = tracing::Span::current();
        let arroy_writers: Result<HashMap<_, _>> = embedders
            .inner_as_ref()
            .iter()
            .map(|(embedder_name, (embedder, _, was_quantized))| {
                let embedder_index = index.embedder_category_id.get(wtxn, embedder_name)?.ok_or(
                    InternalError::DatabaseMissingEntry {
                        db_name: "embedder_category_id",
                        key: None,
                    },
                )?;

                let dimensions = embedder.dimensions();
                let writer = ArroyWrapper::new(vector_arroy, embedder_index, *was_quantized);

                Ok((
                    embedder_index,
                    (embedder_name.as_str(), embedder.as_ref(), writer, dimensions),
                ))
            })
            .collect();

        let mut arroy_writers = arroy_writers?;
        {
            let span = tracing::trace_span!(target: "indexing::write_db", "all");
            let _entered = span.enter();

            for operation in writer_receiver {
                match operation {
                    WriterOperation::DbOperation(db_operation) => {
                        let database = db_operation.database(index);
                        let database_name = db_operation.database_name();
                        match db_operation.entry() {
                            EntryOperation::Delete(e) => match database.delete(wtxn, e.entry()) {
                                Ok(false) => unreachable!("We tried to delete an unknown key"),
                                Ok(_) => (),
                                Err(error) => {
                                    return Err(Error::InternalError(
                                        InternalError::StoreDeletion {
                                            database_name,
                                            key: e.entry().to_owned(),
                                            error,
                                        },
                                    ));
                                }
                            },
                            EntryOperation::Write(e) => {
                                if let Err(error) = database.put(wtxn, e.key(), e.value()) {
                                    return Err(Error::InternalError(InternalError::StorePut {
                                        database_name,
                                        key: e.key().to_owned(),
                                        value_length: e.value().len(),
                                        error,
                                    }));
                                }
                            }
                        }
                    }
                    WriterOperation::ArroyOperation(arroy_operation) => match arroy_operation {
                        ArroyOperation::DeleteVectors { docid } => {
                            for (
                                _embedder_index,
                                (_embedder_name, _embedder, writer, dimensions),
                            ) in &mut arroy_writers
                            {
                                let dimensions = *dimensions;
                                writer.del_items(wtxn, dimensions, docid)?;
                            }
                        }
                        ArroyOperation::SetVectors {
                            docid,
                            embedder_id,
                            embeddings: raw_embeddings,
                        } => {
                            let (_, _, writer, dimensions) = arroy_writers
                                .get(&embedder_id)
                                .expect("requested a missing embedder");

                            let mut embeddings = Embeddings::new(*dimensions);
                            for embedding in raw_embeddings {
                                embeddings.append(embedding).unwrap();
                            }

                            writer.del_items(wtxn, *dimensions, docid)?;
                            writer.add_items(wtxn, docid, &embeddings)?;
                        }
                        ArroyOperation::SetVector { docid, embedder_id, embedding } => {
                            let (_, _, writer, dimensions) = arroy_writers
                                .get(&embedder_id)
                                .expect("requested a missing embedder");
                            writer.del_items(wtxn, *dimensions, docid)?;
                            writer.add_item(wtxn, docid, &embedding)?;
                        }
                        ArroyOperation::Finish { configs } => {
                            let span = tracing::trace_span!(target: "indexing::vectors", parent: &indexer_span, "build");
                            let _entered = span.enter();

                            (indexing_context.send_progress)(Progress::from_step(
                                Step::WritingEmbeddingsToDatabase,
                            ));

                            for (
                                _embedder_index,
                                (_embedder_name, _embedder, writer, dimensions),
                            ) in &mut arroy_writers
                            {
                                let dimensions = *dimensions;
                                writer.build_and_quantize(
                                    wtxn,
                                    &mut rng,
                                    dimensions,
                                    false,
                                    &indexing_context.must_stop_processing,
                                )?;
                            }

                            index.put_embedding_configs(wtxn, configs)?;
                        }
                    },
                }
            }
        }

        (indexing_context.send_progress)(Progress::from_step(Step::WaitingForExtractors));

        let facet_field_ids_delta = extractor_handle.join().unwrap()?;

        (indexing_context.send_progress)(Progress::from_step(Step::PostProcessingFacets));

        if index.facet_search(wtxn)? {
            compute_facet_search_database(index, wtxn, global_fields_ids_map)?;
        }

        compute_facet_level_database(index, wtxn, facet_field_ids_delta)?;

        (indexing_context.send_progress)(Progress::from_step(Step::PostProcessingWords));

        if let Some(prefix_delta) = compute_word_fst(index, wtxn)? {
            compute_prefix_database(index, wtxn, prefix_delta, grenad_parameters)?;
        }

        (indexing_context.send_progress)(Progress::from_step(Step::Finalizing));

        Ok(()) as Result<_>
    })?;

    // required to into_inner the new_fields_ids_map
    drop(fields_ids_map_store);

    let new_fields_ids_map = new_fields_ids_map.into_inner().unwrap();
    index.put_fields_ids_map(wtxn, new_fields_ids_map.as_fields_ids_map())?;

    if let Some(new_primary_key) = new_primary_key {
        index.put_primary_key(wtxn, new_primary_key.name())?;
    }

    // used to update the localized and weighted maps while sharing the update code with the settings pipeline.
    let mut inner_index_settings = InnerIndexSettings::from_index(index, wtxn, Some(embedders))?;
    inner_index_settings.recompute_facets(wtxn, index)?;
    inner_index_settings.recompute_searchables(wtxn, index)?;
    index.put_field_distribution(wtxn, &field_distribution)?;
    index.put_documents_ids(wtxn, &document_ids)?;
    index.set_updated_at(wtxn, &OffsetDateTime::now_utc())?;

    Ok(())
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing::prefix")]
fn compute_prefix_database(
    index: &Index,
    wtxn: &mut RwTxn,
    prefix_delta: PrefixDelta,
    grenad_parameters: GrenadParameters,
) -> Result<()> {
    let PrefixDelta { modified, deleted } = prefix_delta;
    // Compute word prefix docids
    compute_word_prefix_docids(wtxn, index, &modified, &deleted, grenad_parameters)?;
    // Compute exact word prefix docids
    compute_exact_word_prefix_docids(wtxn, index, &modified, &deleted, grenad_parameters)?;
    // Compute word prefix fid docids
    compute_word_prefix_fid_docids(wtxn, index, &modified, &deleted, grenad_parameters)?;
    // Compute word prefix position docids
    compute_word_prefix_position_docids(wtxn, index, &modified, &deleted, grenad_parameters)
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing")]
fn compute_word_fst(index: &Index, wtxn: &mut RwTxn) -> Result<Option<PrefixDelta>> {
    let rtxn = index.read_txn()?;
    let words_fst = index.words_fst(&rtxn)?;
    let mut word_fst_builder = WordFstBuilder::new(&words_fst)?;
    let prefix_settings = index.prefix_settings(&rtxn)?;
    word_fst_builder.with_prefix_settings(prefix_settings);

    let previous_words = index.word_docids.iter(&rtxn)?.remap_data_type::<Bytes>();
    let current_words = index.word_docids.iter(wtxn)?.remap_data_type::<Bytes>();
    for eob in merge_join_by(previous_words, current_words, |lhs, rhs| match (lhs, rhs) {
        (Ok((l, _)), Ok((r, _))) => l.cmp(r),
        (Err(_), _) | (_, Err(_)) => Ordering::Equal,
    }) {
        match eob {
            EitherOrBoth::Both(lhs, rhs) => {
                let (word, lhs_bytes) = lhs?;
                let (_, rhs_bytes) = rhs?;
                if lhs_bytes != rhs_bytes {
                    word_fst_builder.register_word(DelAdd::Addition, word.as_ref())?;
                }
            }
            EitherOrBoth::Left(result) => {
                let (word, _) = result?;
                word_fst_builder.register_word(DelAdd::Deletion, word.as_ref())?;
            }
            EitherOrBoth::Right(result) => {
                let (word, _) = result?;
                word_fst_builder.register_word(DelAdd::Addition, word.as_ref())?;
            }
        }
    }

    let (word_fst_mmap, prefix_data) = word_fst_builder.build(index, &rtxn)?;
    index.main.remap_types::<Str, Bytes>().put(wtxn, WORDS_FST_KEY, &word_fst_mmap)?;
    if let Some(PrefixData { prefixes_fst_mmap, prefix_delta }) = prefix_data {
        index.main.remap_types::<Str, Bytes>().put(
            wtxn,
            WORDS_PREFIXES_FST_KEY,
            &prefixes_fst_mmap,
        )?;
        Ok(Some(prefix_delta))
    } else {
        Ok(None)
    }
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing::facet_search")]
fn compute_facet_search_database(
    index: &Index,
    wtxn: &mut RwTxn,
    global_fields_ids_map: GlobalFieldsIdsMap,
) -> Result<()> {
    let rtxn = index.read_txn()?;
    let localized_attributes_rules = index.localized_attributes_rules(&rtxn)?;
    let mut facet_search_builder = FacetSearchBuilder::new(
        global_fields_ids_map,
        localized_attributes_rules.unwrap_or_default(),
    );

    let previous_facet_id_string_docids = index
        .facet_id_string_docids
        .iter(&rtxn)?
        .remap_data_type::<DecodeIgnore>()
        .filter(|r| r.as_ref().map_or(true, |(k, _)| k.level == 0));
    let current_facet_id_string_docids = index
        .facet_id_string_docids
        .iter(wtxn)?
        .remap_data_type::<DecodeIgnore>()
        .filter(|r| r.as_ref().map_or(true, |(k, _)| k.level == 0));
    for eob in merge_join_by(
        previous_facet_id_string_docids,
        current_facet_id_string_docids,
        |lhs, rhs| match (lhs, rhs) {
            (Ok((l, _)), Ok((r, _))) => l.cmp(r),
            (Err(_), _) | (_, Err(_)) => Ordering::Equal,
        },
    ) {
        match eob {
            EitherOrBoth::Both(lhs, rhs) => {
                let (_, _) = lhs?;
                let (_, _) = rhs?;
            }
            EitherOrBoth::Left(result) => {
                let (key, _) = result?;
                facet_search_builder.register_from_key(DelAdd::Deletion, key)?;
            }
            EitherOrBoth::Right(result) => {
                let (key, _) = result?;
                facet_search_builder.register_from_key(DelAdd::Addition, key)?;
            }
        }
    }

    facet_search_builder.merge_and_write(index, wtxn, &rtxn)
}

#[tracing::instrument(level = "trace", skip_all, target = "indexing::facet_field_ids")]
fn compute_facet_level_database(
    index: &Index,
    wtxn: &mut RwTxn,
    facet_field_ids_delta: FacetFieldIdsDelta,
) -> Result<()> {
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

/// Returns the primary key that has already been set for this index or the
/// one we will guess by searching for the first key that contains "id" as a substring,
/// and whether the primary key changed
/// TODO move this elsewhere
pub fn retrieve_or_guess_primary_key<'a>(
    rtxn: &'a RoTxn<'a>,
    index: &Index,
    new_fields_ids_map: &mut FieldsIdsMap,
    primary_key_from_op: Option<&'a str>,
    first_document: Option<RawMap<'a>>,
) -> Result<StdResult<(PrimaryKey<'a>, bool), UserError>> {
    // make sure that we have a declared primary key, either fetching it from the index or attempting to guess it.

    // do we have an existing declared primary key?
    let (primary_key, has_changed) = if let Some(primary_key_from_db) = index.primary_key(rtxn)? {
        // did we request a primary key in the operation?
        match primary_key_from_op {
            // we did, and it is different from the DB one
            Some(primary_key_from_op) if primary_key_from_op != primary_key_from_db => {
                return Ok(Err(UserError::PrimaryKeyCannotBeChanged(
                    primary_key_from_db.to_string(),
                )));
            }
            _ => (primary_key_from_db, false),
        }
    } else {
        // no primary key in the DB => let's set one
        // did we request a primary key in the operation?
        let primary_key = if let Some(primary_key_from_op) = primary_key_from_op {
            // set primary key from operation
            primary_key_from_op
        } else {
            // guess primary key
            let first_document = match first_document {
                Some(document) => document,
                // previous indexer when no pk is set + we send an empty payload => index_primary_key_no_candidate_found
                None => return Ok(Err(UserError::NoPrimaryKeyCandidateFound)),
            };

            let guesses: Result<Vec<&str>> = first_document
                .keys()
                .filter_map(|name| {
                    let Some(_) = new_fields_ids_map.insert(name) else {
                        return Some(Err(UserError::AttributeLimitReached.into()));
                    };
                    name.to_lowercase().ends_with(DEFAULT_PRIMARY_KEY).then_some(Ok(name))
                })
                .collect();

            let mut guesses = guesses?;

            // sort the keys in lexicographical order, so that fields are always in the same order.
            guesses.sort_unstable();

            match guesses.as_slice() {
                [] => return Ok(Err(UserError::NoPrimaryKeyCandidateFound)),
                [name] => {
                    tracing::info!("Primary key was not specified in index. Inferred to '{name}'");
                    *name
                }
                multiple => {
                    return Ok(Err(UserError::MultiplePrimaryKeyCandidatesFound {
                        candidates: multiple
                            .iter()
                            .map(|candidate| candidate.to_string())
                            .collect(),
                    }))
                }
            }
        };
        (primary_key, true)
    };

    match PrimaryKey::new_or_insert(primary_key, new_fields_ids_map) {
        Ok(primary_key) => Ok(Ok((primary_key, has_changed))),
        Err(err) => Ok(Err(err)),
    }
}

fn request_threads() -> &'static ThreadPoolNoAbort {
    static REQUEST_THREADS: OnceLock<ThreadPoolNoAbort> = OnceLock::new();

    REQUEST_THREADS.get_or_init(|| {
        ThreadPoolNoAbortBuilder::new()
            .num_threads(crate::vector::REQUEST_PARALLELISM)
            .thread_name(|index| format!("embedding-request-{index}"))
            .build()
            .unwrap()
    })
}