//! Store-scoped operation/stage barriers. Compiled only into the unit-test binary.
use crate::{Result, Store};
use std::{
    io::{BufRead, Write},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(crate) enum Action {
    Fail,
    Pause {
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    },
    Process(&'static str),
}

#[derive(Clone, Debug)]
pub(crate) struct Hook {
    pub operation: &'static str,
    pub stage: &'static str,
    pub action: Action,
}

impl Store {
    pub(crate) fn set_test_hook(&self, hook: Option<Hook>) {
        *self.test_hook.lock().expect("test hook mutex") = hook;
    }

    pub(crate) async fn test_boundary(&self, operation: &str, stage: &str) -> Result<()> {
        let hook = self.test_hook.lock().expect("test hook mutex").clone();
        let Some(hook) = hook.filter(|hook| hook.operation == operation && hook.stage == stage)
        else {
            return Ok(());
        };
        match hook.action {
            Action::Fail => {
                Err(std::io::Error::other("injected persistence boundary failure").into())
            }
            Action::Pause { entered, release } => {
                entered.notify_one();
                release.notified().await;
                Ok(())
            }
            Action::Process(label) => {
                tokio::task::spawn_blocking(move || {
                    println!("GW_STAGE:{label}");
                    std::io::stdout().flush()?;
                    let mut line = String::new();
                    std::io::stdin().lock().read_line(&mut line)?;
                    if line.trim() != "RELEASE" {
                        return Err(std::io::Error::other("missing explicit parent release"));
                    }
                    Ok::<_, std::io::Error>(())
                })
                .await??;
                Ok(())
            }
        }
    }
}
