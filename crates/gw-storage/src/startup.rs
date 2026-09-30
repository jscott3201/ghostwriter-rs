//! Startup context and connection initialization, before any Store can be returned.
use crate::{Result, StartupPhase, StorageError};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::time::Duration;
use tokio::time::{Instant, sleep_until, timeout_at};

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);
const INITIAL_BACKOFF: Duration = Duration::from_millis(10);
const MAX_BACKOFF: Duration = Duration::from_millis(100);

pub(crate) struct Startup {
    started: Instant,
    connection_attempts: u32,
    #[cfg(test)]
    hook: Option<StartupHook>,
}
impl Startup {
    pub(crate) fn new(#[cfg(test)] hook: Option<StartupHook>) -> Self {
        Self {
            started: Instant::now(),
            connection_attempts: 0,
            #[cfg(test)]
            hook,
        }
    }
    pub(crate) fn error(
        &self,
        phase: StartupPhase,
        source: impl Into<StorageError>,
    ) -> StorageError {
        StorageError::Startup {
            phase,
            connection_attempts: self.connection_attempts,
            elapsed_ms: self
                .started
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            source: Box::new(source.into()),
        }
    }
    pub(crate) async fn connect(
        &mut self,
        pool_options: SqlitePoolOptions,
        options: SqliteConnectOptions,
    ) -> Result<SqlitePool> {
        let deadline = self.started + CONNECTION_TIMEOUT;
        let mut backoff = INITIAL_BACKOFF;
        let mut last_busy = None;
        loop {
            if Instant::now() >= deadline {
                return Err(self.timeout(last_busy));
            }
            self.connection_attempts += 1;
            // Bound this opener without changing the returned pool's runtime acquire timeout
            // or the SQLite connection's five-second busy timeout. Failed initialization has
            // not entered the migration transaction and its PRAGMAs are safe to repeat.
            let error = match timeout_at(
                deadline,
                pool_options.clone().connect_with(options.clone()),
            )
            .await
            {
                Ok(Ok(pool)) => return Ok(pool),
                Ok(Err(error)) => error,
                Err(_) => return Err(self.timeout(last_busy)),
            };
            #[cfg(test)]
            self.connection_failed(&error).await;
            if !is_busy(&error) {
                return Err(self.error(StartupPhase::Connect, error));
            }
            last_busy = Some(Box::new(error));
            sleep_until((Instant::now() + backoff).min(deadline)).await;
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }

    fn timeout(&self, last_busy: Option<Box<sqlx::Error>>) -> StorageError {
        self.error(
            StartupPhase::Connect,
            StorageError::StartupTimeout { last_busy },
        )
    }
    #[cfg(test)]
    async fn connection_failed(&mut self, error: &sqlx::Error) {
        if let Some(hook) = self.hook.take() {
            let _ = hook.observed.send(StartupObservation {
                phase: StartupPhase::Connect,
                attempt: self.connection_attempts,
                elapsed_ms: self.started.elapsed().as_millis(),
                code: error
                    .as_database_error()
                    .and_then(|error| error.code())
                    .map(|code| code.into_owned()),
            });
            let _ = hook.release.await;
        }
    }
}

fn is_busy(error: &sqlx::Error) -> bool {
    error
        .as_database_error()
        .and_then(|error| error.code())
        .is_some_and(|code| is_busy_code(&code))
}

fn is_busy_code(code: &str) -> bool {
    // SQLite extended result codes retain the primary result in their low byte.
    // SQLITE_LOCKED (6), I/O, permission, and corrupt-file errors are not retryable here.
    code.parse::<u32>().is_ok_and(|code| code & 0xff == 5)
}

#[cfg(test)]
pub(crate) struct StartupHook {
    pub observed: tokio::sync::oneshot::Sender<StartupObservation>,
    pub release: tokio::sync::oneshot::Receiver<()>,
}
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct StartupObservation {
    pub phase: StartupPhase,
    pub attempt: u32,
    pub elapsed_ms: u128,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::is_busy_code;

    #[test]
    fn only_busy_primary_and_extended_codes_are_retried() {
        for code in ["5", "261", "517", "773"] {
            assert!(is_busy_code(code), "{code}");
        }
        for code in ["0", "6", "262", "10", "14", "26", "-5", "busy", ""] {
            assert!(!is_busy_code(code), "{code}");
        }
    }
}
