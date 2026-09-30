//! Await metadata persistence before handing decoded deltas to higher layers.
use crate::{
    DeltaStream, ProviderError,
    delta::parse_chunk,
    metadata,
    observation::ActiveAttempt,
    sse::{SseEvent, decode_events},
};
use futures::StreamExt;
use gw_schema::TransportOutcome;

pub(crate) fn stream(response: reqwest::Response, attempt: ActiveAttempt) -> DeltaStream {
    let status = response.status().as_u16();
    let events = Box::pin(decode_events(response.bytes_stream()));
    Box::pin(futures::stream::unfold(
        Some((events, attempt, None::<ProviderError>)),
        move |state| async move {
            let (mut events, mut attempt, mut parse_error) = state?;
            loop {
                match events.next().await? {
                    Ok(SseEvent::Done) => {
                        let settlement = attempt
                            .settle(
                                TransportOutcome::Complete,
                                Some(status),
                                parse_error.as_ref().map(ToString::to_string),
                            )
                            .await;
                        return match settlement {
                            Err(error) => Some((Err(error), None)),
                            Ok(()) => parse_error.map(|error| (Err(error), None)),
                        };
                    }
                    Ok(SseEvent::Data(payload)) => {
                        // Field extraction is independent of RawChunk and its typed content fields.
                        if let Some(metadata) = metadata::extract_json(payload.as_bytes())
                            && let Err(mut error) = attempt.metadata(metadata).await
                        {
                            if let Some(primary) = parse_error {
                                error = error.with_primary(primary.to_string());
                            }
                            return Some((Err(error), None));
                        }
                        if parse_error.is_some() {
                            continue;
                        }
                        match parse_chunk(&payload) {
                            Ok(delta) => return Some((Ok(delta), Some((events, attempt, None)))),
                            Err(error) => {
                                // Keep ownership of this already-started response. A later usage chunk
                                // remains evidence even though its generated output is unusable.
                                parse_error = Some(ProviderError::Decode(error.to_string()));
                            }
                        }
                    }
                    Err(transport_error) => {
                        let primary = parse_error.unwrap_or(transport_error);
                        let settlement = attempt
                            .settle(
                                TransportOutcome::Failed,
                                Some(status),
                                Some(primary.to_string()),
                            )
                            .await;
                        return Some((Err(settlement.err().unwrap_or(primary)), None));
                    }
                }
            }
        },
    ))
}
