//! One production candidate rendering contract, shared with offline evidence resolution.
use gw_schema::{CotPolicy, Message, TrlFormat};

/// Byte contract used by production judge prompts and the run manifest.
pub const JUDGE_CANDIDATE_RENDER_VERSION: &str = "openai-messages-supervised-v1";

/// Render the complete ordered message sequence with supervised reasoning as a sibling field.
///
/// # Errors
/// Returns the existing format error if messages cannot be represented by the production format.
pub fn render_judge_candidate(messages: &[Message]) -> gw_format::Result<String> {
    gw_format::render(messages, TrlFormat::OpenAiMessages, CotPolicy::Supervised)
}
