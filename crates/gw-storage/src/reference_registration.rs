//! Explicit local operator registration, separate from validation and import.
use crate::{Result, Store, artifact::integrity, now_rfc3339};
use gw_schema::{ReferenceCapture, ValidatedReferenceCatalogue};
use serde::Serialize;

pub(crate) fn id(domain: &str, value: &impl Serialize) -> Result<String> {
    Ok(gw_schema::coding_digest(
        domain,
        &serde_json::to_vec(&serde_json::to_value(value)?)?,
    ))
}
/// Application-owned local registration, loaded only by matching all captured bytes in storage.
/// This does not authenticate an external reviewer or attest a fresh runtime execution.
#[derive(Debug, Clone)]
pub struct RegisteredReferenceCatalogue {
    pub(crate) registration_id: String,
    pub(crate) batch_id: String,
    pub(crate) capture: ReferenceCapture,
    pub(crate) validated: ValidatedReferenceCatalogue,
}
impl RegisteredReferenceCatalogue {
    /// Stable operator registration identity.
    #[must_use]
    pub fn registration_id(&self) -> &str {
        &self.registration_id
    }
    /// Deterministic identity for the one atomic import of this registration.
    #[must_use]
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }
    /// Complete validated captured population, without execution authority.
    #[must_use]
    pub fn population(&self) -> &ValidatedReferenceCatalogue {
        &self.validated
    }
}
fn registration(capture: &ReferenceCapture) -> Result<RegisteredReferenceCatalogue> {
    let validated = capture.validate().map_err(integrity)?;
    let registration_id = id(
        "ghostwriter.reference-registration.v1",
        &validated.catalogue_id(),
    )?;
    let batch_id = id("ghostwriter.reference-batch.v1", &registration_id)?;
    Ok(RegisteredReferenceCatalogue {
        registration_id,
        batch_id,
        capture: capture.clone(),
        validated,
    })
}
impl Store {
    /// Explicit operator acceptance of an exact complete capture. Import never calls this method.
    /// Repeating the same command recognizes the existing registration without changing its facts.
    ///
    /// # Errors
    /// Rejects invalid populations, conflicting identity reuse, or storage failures.
    pub async fn register_reference_catalogue(
        &self,
        capture: &ReferenceCapture,
    ) -> Result<RegisteredReferenceCatalogue> {
        let registered = registration(capture)?;
        let json = serde_json::to_string(capture)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query("INSERT INTO reference_registrations (registration_id,catalogue_id,capture_json,registered_at) VALUES (?,?,?,?) ON CONFLICT(registration_id) DO NOTHING")
            .bind(&registered.registration_id).bind(registered.validated.catalogue_id()).bind(&json)
            .bind(now_rfc3339()).execute(&mut *tx).await?;
        let stored: String = sqlx::query_scalar(
            "SELECT capture_json FROM reference_registrations WHERE registration_id=?",
        )
        .bind(&registered.registration_id)
        .fetch_one(&mut *tx)
        .await?;
        if stored != json {
            return Err(integrity("reference registration capture conflict"));
        }
        tx.commit().await?;
        Ok(registered)
    }
    /// Match every freshly captured byte to an existing operator registration. No auto-registration.
    ///
    /// # Errors
    /// Rejects unregistered, changed or incomplete captures before runtime execution.
    pub async fn registered_reference_catalogue(
        &self,
        capture: &ReferenceCapture,
    ) -> Result<RegisteredReferenceCatalogue> {
        let registered = registration(capture)?;
        let stored: Option<String> = sqlx::query_scalar("SELECT capture_json FROM reference_registrations WHERE registration_id=? AND catalogue_id=?")
            .bind(&registered.registration_id).bind(registered.validated.catalogue_id()).fetch_optional(self.pool()).await?;
        if stored.as_deref() != Some(serde_json::to_string(capture)?.as_str()) {
            return Err(integrity(
                "exact reference capture has no operator registration",
            ));
        }
        Ok(registered)
    }
}
