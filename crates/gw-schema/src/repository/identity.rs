use super::validation::{err, protected, task};
use super::*;
use crate::coding_value::coding_json_digest;
use serde_json::json;

impl RepositoryEpisodeRequest {
    /// Decode one strict bounded request and canonicalize only the changed-path order.
    ///
    /// # Errors
    /// Rejects duplicate or unknown protected fields, unsupported numbers, and invalid declarations.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw = strict_repository_json(bytes)?;
        protected(&raw)?;
        let request: Self =
            serde_json::from_value(raw).map_err(|_| err("invalid repository request fields"))?;
        request.canonicalized()
    }

    /// Return a validated copy with changes sorted by exact path. All trajectory order is retained.
    ///
    /// # Errors
    /// Rejects invalid declarations and requests exceeding the protocol byte bound.
    pub fn canonicalized(&self) -> Result<Self> {
        self.pending()?;
        let mut request = self.clone();
        request
            .candidate
            .delta
            .changes
            .sort_by(|a, b| a.path.cmp(&b.path));
        let bytes = serde_json::to_vec(&request)
            .map_err(|_| err("repository request serialization failed"))?;
        if bytes.len() > REPOSITORY_EPISODE_MAX_BYTES {
            return Err(err("repository request exceeds 32 MiB bound"));
        }
        Ok(request)
    }

    /// Derive separate task, candidate and complete-capture identities from the actual contents.
    ///
    /// Task semantics exclude task label, rights, corpus group and split. Candidate semantics bind
    /// the task, attempt, ordered messages/tools and full supported delta. Capture binds every
    /// declaration, including generation, usage and original publisher reports. These digests prove
    /// internal consistency only; they are neither signatures nor observations of external state.
    ///
    /// # Errors
    /// Rejects invalid or oversized declarations; incomplete supported identities are explicit nulls.
    pub fn identities(&self) -> Result<RepositoryEpisodeIdentities> {
        let request = self.canonicalized()?;
        let pending = request.pending()?;
        let mut task_pending = Vec::new();
        task(&request.task, &mut task_pending)?;
        let value = &request.task;
        let task = task_pending.is_empty().then(|| coding_json_digest("ghostwriter.repository-task.v1", &json!({
            "source": value.source, "problem": value.problem, "repository": value.repository,
            "upstream_base": value.upstream_base, "actor_baseline": value.actor_baseline,
            "environment": value.environment, "private_contract": value.private_contract,
        })));
        let candidate = if pending.is_empty() {
            Some(coding_json_digest(
                "ghostwriter.repository-candidate.v1",
                &json!({
                    "task": task, "attempt": request.candidate.attempt,
                    "messages": request.candidate.messages, "tools": request.candidate.tools,
                    "delta": request.candidate.delta,
                }),
            ))
        } else {
            None
        };
        let capture = coding_json_digest("ghostwriter.repository-capture.v1", &request);
        Ok(RepositoryEpisodeIdentities {
            task,
            candidate,
            capture,
            pending,
        })
    }
}
impl RepositoryEpisodeArtifact {
    /// Decode the saved artifact without granting execution authority or trusting its digests.
    /// Call the independent judge-side verifier to recompute identities and the declaration diagnostic.
    ///
    /// # Errors
    /// Rejects duplicate, unknown protected, malformed, oversized or unsupported fields.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw = strict_repository_json(bytes)?;
        protected(&raw["request"])?;
        let artifact: Self =
            serde_json::from_value(raw).map_err(|_| err("invalid repository artifact fields"))?;
        if artifact.version != 1 {
            return Err(err("unsupported repository artifact version"));
        }
        artifact.request.pending()?;
        Ok(artifact)
    }
}
