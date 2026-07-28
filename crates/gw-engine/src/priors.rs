//! Run-scoped embedding priors used by the user-turn diversity gate.
//!
//! One canonical corpus owns every vector exactly once. Common gate snapshots clone its [`Arc`] in
//! O(1). If an item with admitted vectors gates again, exclusion is uniformly derived as the current
//! corpus minus that seed item's vectors. Only its linear index set is cached; the filtered vectors
//! are temporary and never retained. Appends invalidate those lazy index caches. Corpus growth is
//! intentionally unbounded for the run because dedup requires the complete admitted-turn history.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

use gw_generate::Embedder;
use gw_schema::{Content, ContentPart, Role, TrainingRecord};

#[derive(Default)]
pub(crate) struct PriorState {
    all: Arc<Vec<Vec<f32>>>,
    item_ids: Vec<String>,
    items_with_vectors: HashSet<String>,
    exclusion_indices: HashMap<String, Arc<Vec<usize>>>,
}

/// Shared, run-scoped embedding vectors.
pub(crate) type Priors = Arc<RwLock<PriorState>>;

pub(crate) fn new() -> Priors {
    Arc::new(RwLock::new(PriorState::default()))
}

/// Content-independent identity shared by all siblings/retries of one seed item.
pub(crate) fn item_id(run_id: &str, shard: i64, seed: i64) -> String {
    format!("{run_id}-s{shard}-seed{seed}")
}

pub(crate) fn record_item_id(record_id: &str) -> Option<String> {
    record_id
        .rsplit_once("-a")
        .map(|(prefix, _)| prefix.to_string())
}

pub(crate) fn snapshot(priors: &Priors, item_id: &str) -> Arc<Vec<Vec<f32>>> {
    {
        let state = priors
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.items_with_vectors.contains(item_id) {
            return Arc::clone(&state.all);
        }
    }

    let (all, excluded) = {
        let mut state = priors
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let excluded = if let Some(cached) = state.exclusion_indices.get(item_id) {
            Arc::clone(cached)
        } else {
            let indices = Arc::new(
                state
                    .item_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(index, owner)| (owner == item_id).then_some(index))
                    .collect(),
            );
            state
                .exclusion_indices
                .insert(item_id.to_string(), Arc::clone(&indices));
            indices
        };
        (Arc::clone(&state.all), excluded)
    };
    Arc::new(
        all.iter()
            .enumerate()
            .filter(|(index, _)| excluded.binary_search(index).is_err())
            .map(|(_, vector)| vector.clone())
            .collect(),
    )
}

pub(crate) fn replace(priors: &Priors, tagged: Vec<(String, Vec<f32>)>) {
    let (item_ids, vectors): (Vec<_>, Vec<_>) = tagged.into_iter().unzip();
    let items_with_vectors = item_ids.iter().cloned().collect();
    *priors
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = PriorState {
        all: Arc::new(vectors),
        item_ids,
        items_with_vectors,
        exclusion_indices: HashMap::new(),
    };
}

fn insert(priors: &Priors, item_id: String, vector: Vec<f32>) {
    let mut state = priors
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Arc::make_mut(&mut state.all).push(vector);
    state.item_ids.push(item_id.clone());
    state.items_with_vectors.insert(item_id);
    state.exclusion_indices.clear();
}

pub(crate) fn append_record(priors: &Priors, embedder: &dyn Embedder, record: &TrainingRecord) {
    let Some(text) = user_turn_text(record) else {
        tracing::warn!(record_id = %record.record_id, "admitted record has no textual user turn; skipping embedding prior");
        return;
    };
    let Some(item_id) = record_item_id(&record.record_id) else {
        tracing::warn!(record_id = %record.record_id, "admitted record has no seed-item identity; skipping embedding prior");
        return;
    };
    // Never hold the lock across the blocking embed call.
    match embedder.embed(&text) {
        Ok(vector) => insert(priors, item_id, vector),
        Err(error) => tracing::warn!(
            record_id = %record.record_id,
            %error,
            "failed to embed admitted user turn; skipping embedding prior"
        ),
    }
}

pub(crate) fn user_turn_text(record: &TrainingRecord) -> Option<String> {
    user_turn_text_from_messages(&record.messages)
}

fn user_turn_text_from_messages(messages: &[gw_schema::Message]) -> Option<String> {
    let text = messages
        .iter()
        .find(|message| message.role == Role::User)
        .map(|message| match &message.content {
            Content::Text(text) => text.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        })?;
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
mod tests {
    use gw_schema::Message;

    use super::*;

    fn user_parts(parts: Vec<ContentPart>) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: Content::Parts(parts),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            name: None,
        }]
    }

    #[test]
    fn user_turn_text_concatenates_only_text_parts() {
        let messages = user_parts(vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::ImageUrl {
                image_url: "image".into(),
            },
            ContentPart::Text { text: "b".into() },
        ]);
        assert_eq!(
            user_turn_text_from_messages(&messages).as_deref(),
            Some("ab")
        );
    }

    #[test]
    fn image_only_user_turn_has_no_embeddable_text() {
        let messages = user_parts(vec![ContentPart::ImageUrl {
            image_url: "image".into(),
        }]);
        assert_eq!(user_turn_text_from_messages(&messages), None);
    }

    #[test]
    fn same_item_snapshot_excludes_only_own_vectors() {
        let priors = new();
        replace(
            &priors,
            vec![
                ("a".into(), vec![1.0]),
                ("a".into(), vec![2.0]),
                ("b".into(), vec![3.0]),
            ],
        );
        assert_eq!(snapshot(&priors, "a").as_ref(), &vec![vec![3.0]]);
        assert_eq!(snapshot(&priors, "c").len(), 3);
    }

    #[test]
    fn common_snapshots_share_one_linear_corpus() {
        let priors = new();
        replace(
            &priors,
            (0..100)
                .map(|index| (format!("item-{index}"), vec![index as f32; 8]))
                .collect(),
        );
        let first = snapshot(&priors, "new-item");
        let second = snapshot(&priors, "another-new-item");
        assert!(Arc::ptr_eq(&first, &second));
        let state = priors.read().unwrap();
        assert_eq!(state.all.len(), 100);
        assert_eq!(state.item_ids.len(), 100);
        assert!(state.exclusion_indices.is_empty());
    }

    #[test]
    fn exclusion_cache_retains_indices_not_vector_corpora_and_invalidates() {
        let priors = new();
        replace(
            &priors,
            vec![("a".into(), vec![1.0]), ("b".into(), vec![2.0])],
        );
        let _ = snapshot(&priors, "a");
        {
            let state = priors.read().unwrap();
            assert_eq!(state.exclusion_indices["a"].as_ref(), &vec![0]);
        }
        insert(&priors, "c".into(), vec![3.0]);
        assert!(priors.read().unwrap().exclusion_indices.is_empty());
        assert_eq!(priors.read().unwrap().all.len(), 3);
    }

    #[test]
    fn record_item_identity_strips_attempt_and_completion_suffix() {
        assert_eq!(
            record_item_id("run-a-s2-seed7-a1-c3").as_deref(),
            Some("run-a-s2-seed7")
        );
    }
}
