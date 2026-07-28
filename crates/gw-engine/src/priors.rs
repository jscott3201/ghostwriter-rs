//! Run-scoped embedding priors used by the user-turn diversity gate.
//!
//! Gate snapshots clone an [`Arc`] in O(1). Appends use copy-on-write and retain an item's
//! pre-admission corpus so siblings never dedup against their own winner. Corpus growth is
//! intentionally unbounded for the run: dedup requires the complete admitted-turn history.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use gw_generate::Embedder;
use gw_schema::{Content, ContentPart, Role, TrainingRecord};

#[derive(Default)]
pub(crate) struct PriorState {
    all: Arc<Vec<Vec<f32>>>,
    without_item: HashMap<String, Arc<Vec<Vec<f32>>>>,
}

/// Shared, run-scoped embedding vectors.
pub(crate) type Priors = Arc<RwLock<PriorState>>;

pub(crate) fn new() -> Priors {
    Arc::new(RwLock::new(PriorState::default()))
}

pub(crate) fn snapshot(priors: &Priors, item_id: &str) -> Arc<Vec<Vec<f32>>> {
    let state = priors
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state
        .without_item
        .get(item_id)
        .cloned()
        .unwrap_or_else(|| Arc::clone(&state.all))
}

pub(crate) fn replace(priors: &Priors, tagged: Vec<(String, Vec<f32>)>) {
    let all = Arc::new(
        tagged
            .iter()
            .map(|(_, vector)| vector.clone())
            .collect::<Vec<_>>(),
    );
    let mut without_item = HashMap::new();
    for (seed, _) in &tagged {
        without_item.entry(seed.clone()).or_insert_with(|| {
            Arc::new(
                tagged
                    .iter()
                    .filter(|(other, _)| other != seed)
                    .map(|(_, vector)| vector.clone())
                    .collect(),
            )
        });
    }
    *priors
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = PriorState { all, without_item };
}

pub(crate) fn append_record(priors: &Priors, embedder: &dyn Embedder, record: &TrainingRecord) {
    let Some(text) = user_turn_text(record) else {
        tracing::warn!(record_id = %record.record_id, "admitted record has no textual user turn; skipping embedding prior");
        return;
    };
    let Some(item_id) = record.generation.sibling_group_id.clone() else {
        tracing::warn!(record_id = %record.record_id, "admitted record has no item identity; skipping embedding prior");
        return;
    };
    // Never hold the lock across the blocking embed call.
    match embedder.embed(&text) {
        Ok(vector) => {
            let mut state = priors
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let before = Arc::clone(&state.all);
            state.without_item.entry(item_id).or_insert(before);
            Arc::make_mut(&mut state.all).push(vector);
        }
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
    fn snapshot_excludes_vectors_from_same_item() {
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
    fn repeated_snapshots_share_the_same_corpus_allocation() {
        let priors = new();
        replace(
            &priors,
            vec![("a".into(), vec![1.0]), ("b".into(), vec![2.0])],
        );
        let first = snapshot(&priors, "other");
        let second = snapshot(&priors, "other");
        assert!(Arc::ptr_eq(&first, &second));
    }
}
