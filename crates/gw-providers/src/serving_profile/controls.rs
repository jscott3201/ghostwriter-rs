use super::*;
use gw_schema::ModelOperation;
use serde_json::json;
use std::collections::BTreeSet;

pub(super) fn apply(
    behavior: &ProfileBehavior,
    operation: ModelOperation,
    controls: &[RequestedControl],
    body: &mut Value,
) -> Result<Vec<ControlDegradation>> {
    let mut state = Controls {
        behavior,
        operation,
        body,
        seen: BTreeSet::new(),
        degradations: Vec::new(),
        reasoning: false,
    };
    for requested in controls {
        state.apply(&requested.value, requested.required)?;
    }
    state.degradations.sort_by_key(|v| v.control);
    Ok(state.degradations)
}
struct Controls<'a> {
    behavior: &'a ProfileBehavior,
    operation: ModelOperation,
    body: &'a mut Value,
    seen: BTreeSet<ProfileControl>,
    degradations: Vec<ControlDegradation>,
    reasoning: bool,
}
impl Controls<'_> {
    fn supported(
        &mut self,
        control: ProfileControl,
        required: bool,
        value_supported: bool,
    ) -> Result<bool> {
        if !self.seen.insert(control) {
            return Err(ProfileError::InvalidRequest);
        }
        let operation_supports = match control {
            ProfileControl::Temperature
            | ProfileControl::TopP
            | ProfileControl::Seed
            | ProfileControl::MaxOutputTokens
            | ProfileControl::Usage => self.operation != ModelOperation::Embedding,
            _ => self.operation == ModelOperation::ChatCompletion,
        };
        let reason = if !self.behavior.supports(control) || !operation_supports {
            Some(DegradationReason::Unsupported)
        } else if !value_supported {
            Some(DegradationReason::UnsupportedValue)
        } else {
            None
        };
        if let Some(reason) = reason {
            if required {
                return Err(ProfileError::UnsupportedRequiredControl(control));
            }
            self.degradations
                .push(ControlDegradation { control, reason });
            Ok(false)
        } else {
            Ok(true)
        }
    }
    fn field(
        &mut self,
        control: ProfileControl,
        required: bool,
        name: &str,
        value: Value,
    ) -> Result<()> {
        if self.supported(control, required, true)? {
            self.body[name] = value;
        }
        Ok(())
    }
    fn apply(&mut self, value: &ControlValue, required: bool) -> Result<()> {
        match value {
            ControlValue::Sampling(sampling) => {
                if !sampling.temperature.is_finite()
                    || !(0.0..=2.0).contains(&sampling.temperature)
                    || sampling
                        .top_p
                        .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v) || v == 0.0)
                {
                    return Err(ProfileError::InvalidRequest);
                }
                self.field(
                    ProfileControl::Temperature,
                    required,
                    "temperature",
                    json!(sampling.temperature),
                )?;
                if let Some(value) = sampling.top_p {
                    self.field(ProfileControl::TopP, required, "top_p", json!(value))?;
                }
                if let Some(value) = sampling.seed {
                    self.field(ProfileControl::Seed, required, "seed", json!(value))?;
                }
            }
            ControlValue::MaxOutputTokens(value) => {
                if *value == 0 {
                    return Err(ProfileError::InvalidRequest);
                }
                self.field(
                    ProfileControl::MaxOutputTokens,
                    required,
                    "max_tokens",
                    json!(value),
                )?;
            }
            ControlValue::ReasoningEffort(value) => {
                self.unique_reasoning()?;
                if !text_valid(value) {
                    return Err(ProfileError::InvalidRequest);
                }
                if self.supported(
                    ProfileControl::ReasoningEffort,
                    required,
                    self.behavior.reasoning_efforts.contains(value),
                )? {
                    match self.behavior.dialect {
                        ServingDialect::OpenRouterV1 => {
                            self.body["reasoning"] = json!({"effort":value})
                        }
                        ServingDialect::VllmV1 => self.body["reasoning_effort"] = json!(value),
                    }
                }
            }
            ControlValue::ReasoningBudget(value) => {
                self.unique_reasoning()?;
                if *value == 0 {
                    return Err(ProfileError::InvalidRequest);
                }
                if self.supported(ProfileControl::ReasoningBudget, required, true)? {
                    self.body["reasoning"] = json!({"max_tokens":value});
                }
            }
            ControlValue::EnableThinking(value) => self.template(
                ProfileControl::EnableThinking,
                required,
                "enable_thinking",
                *value,
            )?,
            ControlValue::PreserveThinking(value) => self.template(
                ProfileControl::PreserveThinking,
                required,
                "preserve_thinking",
                *value,
            )?,
            ControlValue::ClearThinking(value) => self.template(
                ProfileControl::ClearThinking,
                required,
                "clear_thinking",
                *value,
            )?,
            ControlValue::JsonSchema { name, schema } => {
                if !text_valid(name) || !schema.is_object() {
                    return Err(ProfileError::InvalidRequest);
                }
                if self.supported(ProfileControl::JsonSchema, required, true)? {
                    self.body["response_format"] = json!({"type":"json_schema","json_schema":{"name":name,"strict":true,"schema":schema}});
                }
            }
            ControlValue::Logprobs(count) => {
                if *count > 20 {
                    return Err(ProfileError::InvalidRequest);
                }
                if self.supported(ProfileControl::Logprobs, required, true)? {
                    self.body["logprobs"] = json!(true);
                    self.body["top_logprobs"] = json!(count);
                }
            }
            ControlValue::Usage => {
                if self.supported(ProfileControl::Usage, required, true)? {
                    match self.behavior.dialect {
                        ServingDialect::OpenRouterV1 => {
                            self.body["usage"] = json!({"include":true})
                        }
                        ServingDialect::VllmV1 => {
                            self.body["stream_options"] = json!({"include_usage":true})
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn template(
        &mut self,
        control: ProfileControl,
        required: bool,
        name: &str,
        value: bool,
    ) -> Result<()> {
        if self.supported(control, required, true)? {
            if self.body.get("chat_template_kwargs").is_none() {
                self.body["chat_template_kwargs"] = json!({});
            }
            self.body["chat_template_kwargs"][name] = json!(value);
        }
        Ok(())
    }
    fn unique_reasoning(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.reasoning, true) {
            Err(ProfileError::InvalidRequest)
        } else {
            Ok(())
        }
    }
}
