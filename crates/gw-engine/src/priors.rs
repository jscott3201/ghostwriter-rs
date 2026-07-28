//! Run-scoped embedding priors used by the user-turn diversity gate.

use std::sync::{Arc, RwLock};

use gw_generate::Embedder;
use gw_schema::{Content, ContentPart, Role, TrainingRecord};

/// Shared, run-scoped embedding vectors.
pub(crate) type Priors = Arc<RwLock<Vec<Vec<f32>>>>;

pub(crate) fn new() -> Priors {
    Arc::new(RwLock::new(Vec::new()))
}

pub(crate) fn snapshot(priors: &Priors) -> Vec<Vec<f32>> {
    priors
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

pub(crate) fn replace(priors: &Priors, vectors: Vec<Vec<f32>>) {
    *priors
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = vectors;
}

pub(crate) fn append_record(priors: &Priors, embedder: &dyn Embedder, record: &TrainingRecord) {
    let Some(text) = user_turn_text(record) else {
        tracing::warn!(record_id = %record.record_id, "admitted record has no user turn; skipping embedding prior");
        return;
    };
    // Never hold the lock across the blocking embed call.
    match embedder.embed(&text) {
        Ok(vector) => priors
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(vector),
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
    messages
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
        })
}

#[cfg(test)]
mod tests {
    use gw_schema::Message;

    use super::*;

    #[test]
    fn user_turn_text_concatenates_only_text_parts() {
        let messages = vec![Message {
            role: Role::User,
            content: Content::Parts(vec![
                ContentPart::Text { text: "a".into() },
                ContentPart::ImageUrl {
                    image_url: "image".into(),
                },
                ContentPart::Text { text: "b".into() },
            ]),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            name: None,
        }];
        assert_eq!(
            user_turn_text_from_messages(&messages).as_deref(),
            Some("ab")
        );
    }
}
