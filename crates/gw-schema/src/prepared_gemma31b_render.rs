//! Independent native rendering and ownership for the qualified serial subset.
use crate::{CotPolicy, PreparedSftRenderControls, PreparedSftSpan};
use serde_json::Value;
const Q: &str = "<|\"|>";
type Result<T> = std::result::Result<T, &'static str>;

fn string(value: &Value) -> Result<&str> {
    value.as_str().ok_or("expected text in serial source")
}
fn ordered(value: &Value) -> Result<Vec<(&String, &Value)>> {
    let mut values: Vec<_> = value
        .as_object()
        .ok_or("expected serial object")?
        .iter()
        .collect();
    values.sort_by_key(|(key, _)| key.to_lowercase());
    Ok(values)
}
pub(super) fn argument(value: &Value, escaped: bool) -> Result<String> {
    Ok(match value {
        Value::String(s) => format!("{Q}{s}{Q}"),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|v| argument(v, escaped))
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Value::Object(_) => format!(
            "{{{}}}",
            ordered(value)?
                .into_iter()
                .map(|(k, v)| {
                    Ok(format!(
                        "{}:{}",
                        if escaped {
                            format!("{Q}{k}{Q}")
                        } else {
                            k.clone()
                        },
                        argument(v, escaped)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Value::Number(number) if number.is_f64() => {
            let number = number.as_f64().ok_or("invalid argument float")?;
            // Python/Jinja spellings for finite decimal floats; scientific spellings are
            // rejected by exact official replay until explicitly qualified.
            if !number.is_finite() || number.abs() >= 1e16 || (number != 0.0 && number.abs() < 1e-4)
            {
                return Err("unsupported scientific argument float");
            }
            let rendered = number.to_string();
            if rendered.contains('.') {
                rendered
            } else {
                format!("{rendered}.0")
            }
        }
        other => other.to_string(),
    })
}
fn properties(value: &Value) -> Result<String> {
    ordered(value)?
        .into_iter()
        .map(|(key, value)| Ok(format!("{key}:{{{}}}", property(value)?)))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.join(","))
}
fn property(value: &Value) -> Result<String> {
    let mut parts = Vec::new();
    if value["description"].as_str().is_some_and(|s| !s.is_empty()) {
        parts.push(format!(
            "description:{}",
            argument(&value["description"], false)?
        ));
    }
    let kind = string(&value["type"])?;
    if kind == "string" && value["enum"].as_array().is_some_and(|v| !v.is_empty()) {
        parts.push(format!("enum:{}", argument(&value["enum"], true)?));
    }
    if kind == "array" {
        let mut items = Vec::new();
        for (key, value) in ordered(&value["items"])? {
            let rendered = match key.as_str() {
                "properties" => format!("{{{}}}", properties(value)?),
                "type" => format!("{Q}{}{Q}", string(value)?.to_uppercase()),
                _ => argument(value, true)?,
            };
            items.push(format!("{key}:{rendered}"));
        }
        parts.push(format!("items:{{{}}}", items.join(",")));
    }
    if kind == "object" {
        parts.push(format!(
            "properties:{{{}}}",
            properties(&value["properties"])?
        ));
        if value["required"].as_array().is_some_and(|v| !v.is_empty()) {
            parts.push(format!("required:{}", argument(&value["required"], false)?));
        }
    }
    parts.push(format!("type:{Q}{}{Q}", kind.to_uppercase()));
    Ok(parts.join(","))
}
fn definition(tool: &Value) -> Result<String> {
    let f = &tool["function"];
    let params = &f["parameters"];
    let mut parts = Vec::new();
    if params["properties"]
        .as_object()
        .is_some_and(|p| !p.is_empty())
    {
        parts.push(format!(
            "properties:{{{}}}",
            properties(&params["properties"])?
        ));
    }
    if params["required"].as_array().is_some_and(|v| !v.is_empty()) {
        parts.push(format!(
            "required:{}",
            argument(&params["required"], false)?
        ));
    }
    parts.push(format!(
        "type:{Q}{}{Q}",
        string(&params["type"])?.to_uppercase()
    ));
    Ok(format!(
        "<|tool>declaration:{}{{description:{Q}{}{Q},parameters:{{{}}}}}<tool|>",
        string(&f["name"])?,
        f["description"].as_str().unwrap_or(""),
        parts.join(",")
    ))
}
#[derive(Default)]
struct Ledger {
    rendered: String,
    spans: Vec<PreparedSftSpan>,
    position: u64,
}
impl Ledger {
    fn emit(&mut self, text: &str, kind: &str, supervised: bool, index: usize) {
        if text.is_empty() {
            return;
        }
        let end = self.position + text.chars().count() as u64;
        self.spans.push(PreparedSftSpan {
            start: self.position,
            end,
            kind: kind.into(),
            supervised,
            message_index: index as u64,
        });
        self.position = end;
        self.rendered.push_str(text);
    }
}
fn trim(value: &str) -> &str {
    value.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}'))
}
pub(super) fn render(
    messages: &[Value],
    tools: &[Value],
    cot: CotPolicy,
    controls: &PreparedSftRenderControls,
) -> Result<(String, Vec<PreparedSftSpan>)> {
    let mut l = Ledger::default();
    l.emit("<bos>", "header", false, 0);
    let system = messages[0]["role"] == "system";
    if system || !tools.is_empty() || controls.enable_thinking {
        l.emit("<|turn>system\n", "header", false, 0);
        if controls.enable_thinking {
            l.emit("<|think|>\n", "header", false, 0);
        }
        if system {
            l.emit(trim(string(&messages[0]["content"])?), "context", false, 0);
        }
        for tool in tools {
            l.emit(&definition(tool)?, "definition", false, 0);
        }
        l.emit("<turn|>", "end", false, 0);
        l.emit("\n", "separator", false, 0);
    }
    let last_user = messages
        .iter()
        .rposition(|m| m["role"] == "user")
        .ok_or("serial source requires user")?;
    let target = messages.len() - 1;
    let mut previous = "";
    for (index, m) in messages.iter().enumerate().skip(usize::from(system)) {
        let role = string(&m["role"])?;
        if role == "tool" {
            let name = string(&messages[index - 1]["tool_calls"][0]["function"]["name"])?;
            l.emit(
                &format!(
                    "<|tool_response>response:{name}{{value:{}}}<tool_response|>",
                    argument(&m["content"], false)?
                ),
                "observation",
                false,
                index,
            );
            continue;
        }
        if role != "assistant" || previous != "assistant" {
            l.emit(
                &format!(
                    "<|turn>{}\n",
                    if role == "assistant" { "model" } else { role }
                ),
                "header",
                false,
                index,
            );
        }
        let calls = m["tool_calls"].as_array();
        if cot != CotPolicy::Stripped
            && (index > last_user || (controls.preserve_thinking == Some(true) && calls.is_some()))
            && let Some(reasoning) = m["reasoning"].as_str().filter(|s| !s.is_empty())
        {
            let supervised = index == target && cot == CotPolicy::Supervised;
            l.emit(
                "<|channel>thought\n",
                "reasoning_wrapper",
                supervised,
                index,
            );
            l.emit(reasoning, "reasoning", supervised, index);
            l.emit("\n<channel|>", "reasoning_wrapper", supervised, index);
        }
        if let Some(calls) = calls {
            let f = &calls[0]["function"];
            l.emit("<|tool_call>", "call_wrapper", index == target, index);
            l.emit(
                &format!(
                    "call:{}{}",
                    string(&f["name"])?,
                    argument(&f["arguments"], false)?
                ),
                "call",
                index == target,
                index,
            );
            l.emit("<tool_call|>", "call_wrapper", index == target, index);
            if index == target {
                l.emit("<|tool_response>", "handoff", true, index);
            }
        } else {
            l.emit(
                trim(string(&m["content"])?),
                if index == target { "answer" } else { "context" },
                index == target,
                index,
            );
            l.emit("<turn|>", "end", index == target, index);
            l.emit("\n", "separator", false, index);
        }
        previous = role;
    }
    Ok((l.rendered, l.spans))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_definitions_match_independent_literal_contract() {
        let cases: Value = serde_json::from_str(include_str!(
            "../../../adapters/trl/tests/gemma31b/definition_cases.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            assert_eq!(
                definition(&case["tool"]).unwrap(),
                case["rendered"].as_str().unwrap()
            );
        }
    }
}
