//! Full captured input identity and effective execution settings, without model execution.
mod common;
use common::*;
use gw_engine::{
    AreaConfig, CapturedSeedPlan, Engine, EventSink, InMemorySeedSource, SeedItem, SeedSource,
};
use gw_generate::UserTurnCandidate;
use gw_schema::AdmissionIntent;
use gw_storage::Store;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;
fn digest(candidate: UserTurnCandidate) -> String {
    CapturedSeedPlan::capture(&InMemorySeedSource::new(vec![candidate], 1))
        .unwrap()
        .identity()
        .content_hash
        .clone()
}
fn rich_candidate() -> UserTurnCandidate {
    serde_json::from_value(json!({
        "message":{"role":"user","content":[{"type":"text","text":" q "},{"type":"image_url","image_url":"https://invalid.test/image"},{"type":"input_audio","audio_url":"https://invalid.test/audio","format":"wav"}],
          "reasoning":" thoughts ","reasoning_details":[{"type":"reasoning.text","text":"text","signature":"sig","id":"id","format":"native","index":0},{"type":"reasoning.summary","summary":"summary","id":"sum","format":"native","index":1},{"type":"reasoning.encrypted","data":"cipher","id":"enc","format":"native","index":2}],
          "tool_calls":[{"id":"call-a","function":{"name":"tool","arguments":{"x":1},"raw_arguments":"{ \"x\":1 }"}},{"id":"call-b","function":{"name":"tool","arguments":{"x":2}}}],"tool_call_id":"result-a","name":"tool-result"},
        "seed":{"persona":"persona","taxonomy_node":"node","prompt_template_id":"template","difficulty":"hard"},
        "contract":{"answer_policy":"absent","execution_policy":"absent","kind":"set_match","oracle":{"oracle":"sandbox_execution","tool_or_sql":"SELECT 1","expected":"NaN"}},
        "answerable":true,"difficulty_targeted":true,"in_scope":true
    })).unwrap()
}
#[test]
fn every_nested_candidate_field_changes_identity_without_fetching_media() {
    let base = rich_candidate();
    let before = digest(base.clone());
    let value = serde_json::to_value(base).unwrap();
    let paths = [
        "/message/role",
        "/message/content/0/text",
        "/message/content/1/image_url",
        "/message/content/2/audio_url",
        "/message/content/2/format",
        "/message/reasoning",
        "/message/reasoning_details/0/text",
        "/message/reasoning_details/0/signature",
        "/message/reasoning_details/0/id",
        "/message/reasoning_details/0/format",
        "/message/reasoning_details/0/index",
        "/message/reasoning_details/1/summary",
        "/message/reasoning_details/1/id",
        "/message/reasoning_details/1/format",
        "/message/reasoning_details/1/index",
        "/message/reasoning_details/2/data",
        "/message/reasoning_details/2/id",
        "/message/reasoning_details/2/format",
        "/message/reasoning_details/2/index",
        "/message/tool_calls/0/id",
        "/message/tool_calls/0/function/name",
        "/message/tool_calls/0/function/arguments/x",
        "/message/tool_calls/0/function/raw_arguments",
        "/message/tool_call_id",
        "/message/name",
        "/seed/persona",
        "/seed/taxonomy_node",
        "/seed/prompt_template_id",
        "/seed/difficulty",
        "/contract/kind",
        "/contract/oracle/tool_or_sql",
        "/contract/oracle/expected",
        "/answerable",
        "/difficulty_targeted",
        "/in_scope",
    ];
    for path in paths {
        let mut changed = value.clone();
        let field = changed.pointer_mut(path).unwrap();
        *field = match path {
            "/message/role" => json!("system"),
            "/contract/kind" => json!("schema_shape"),
            _ => match field {
                Value::String(s) => json!(format!("{s} ")),
                Value::Bool(b) => json!(!*b),
                Value::Number(n) => json!(n.as_u64().unwrap() + 1),
                _ => panic!("unexpected {path}"),
            },
        };
        let candidate: UserTurnCandidate = serde_json::from_value(changed).unwrap();
        assert_ne!(digest(candidate), before, "omitted field {path}");
    }
    for path in [
        "/message/content",
        "/message/tool_calls",
        "/message/reasoning_details",
    ] {
        let mut changed = value.clone();
        changed
            .pointer_mut(path)
            .unwrap()
            .as_array_mut()
            .unwrap()
            .reverse();
        assert_ne!(
            digest(serde_json::from_value(changed).unwrap()),
            before,
            "array order {path}"
        );
    }
}
#[test]
fn null_empty_whitespace_oracle_variants_and_seed_options_remain_distinct() {
    let base = serde_json::to_value(good_candidate("q")).unwrap();
    for (path, values) in [
        (
            "/message/content",
            vec![
                Value::Null,
                json!(""),
                json!(" "),
                json!([]),
                json!([{ "type":"text","text":""}]),
            ],
        ),
        (
            "/message/reasoning",
            vec![Value::Null, json!(""), json!(" ")],
        ),
        ("/message/reasoning_details", vec![Value::Null, json!([])]),
        ("/message/tool_calls", vec![Value::Null, json!([])]),
        (
            "/seed/difficulty",
            vec![Value::Null, json!(""), json!("easy")],
        ),
        (
            "/contract/oracle",
            vec![
                json!({"oracle":"none"}),
                json!({"oracle":"literal","expected":"NaN"}),
                json!({"oracle":"literal","expected":"inf"}),
                json!({"oracle":"literal","expected":""}),
                json!({"oracle":"refusal_policy","policy_id":"p"}),
                json!({"oracle":"sandbox_execution","tool_or_sql":"select 1","expected":null}),
                json!({"oracle":"sandbox_execution","tool_or_sql":"select 1","expected":""}),
            ],
        ),
    ] {
        let mut hashes = std::collections::HashSet::new();
        for value in values {
            let mut changed = base.clone();
            let (parent, key) = path.rsplit_once('/').unwrap();
            changed
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert(key.into(), value);
            assert!(
                hashes.insert(digest(serde_json::from_value(changed).unwrap())),
                "collapsed {path}"
            );
        }
    }
}
struct Plan {
    count: usize,
    shards: Vec<Vec<SeedItem>>,
}
impl SeedSource for Plan {
    fn shard_count(&self) -> usize {
        self.count
    }
    fn items_for_shard(&self, s: i64) -> Vec<SeedItem> {
        self.shards[s as usize].clone()
    }
}
fn item(seed: i64, offset: u64) -> SeedItem {
    SeedItem {
        seed,
        offset,
        candidate: good_candidate("q"),
    }
}
#[test]
fn offsets_and_seeds_protect_record_cursor_identity_including_empty_shards() {
    for items in [
        vec![item(0, 1)],
        vec![item(0, u64::MAX)],
        vec![item(0, 0), item(0, 1)],
        vec![item(0, 0), item(1, 0)],
    ] {
        assert!(
            CapturedSeedPlan::capture(&Plan {
                count: 1,
                shards: vec![items]
            })
            .is_err()
        );
    }
    let negative = CapturedSeedPlan::capture(&Plan {
        count: 3,
        shards: vec![vec![item(-1, 0)], vec![], vec![item(-1, 0)]],
    })
    .unwrap();
    assert_eq!(negative.identity().shard_items, vec![1, 0, 1]);
    let zero = CapturedSeedPlan::capture(&Plan {
        count: 0,
        shards: vec![vec![item(0, 0)]],
    })
    .unwrap();
    let one = CapturedSeedPlan::capture(&Plan {
        count: 1,
        shards: vec![vec![item(0, 0)]],
    })
    .unwrap();
    assert_eq!(zero, one);
    let extra_empty = CapturedSeedPlan::capture(&Plan {
        count: 2,
        shards: vec![vec![item(0, 0)], vec![]],
    })
    .unwrap();
    assert_ne!(one.identity(), extra_empty.identity());
    assert!(
        CapturedSeedPlan::capture(&Plan {
            count: usize::MAX,
            shards: vec![]
        })
        .is_err()
    );
    let wrapped = CapturedSeedPlan::capture(&InMemorySeedSource::with_base_seed(
        vec![good_candidate("a"), good_candidate("b")],
        1,
        i64::MAX,
    ))
    .unwrap();
    assert_eq!(wrapped.shards()[0][1].seed, i64::MIN);
    let changed_seed = CapturedSeedPlan::capture(&Plan {
        count: 1,
        shards: vec![vec![item(-1, 0)]],
    })
    .unwrap();
    assert_ne!(one.identity(), changed_seed.identity());
}
struct ChangingSource {
    counts: AtomicUsize,
    queries: [AtomicUsize; 2],
}
impl SeedSource for ChangingSource {
    fn shard_count(&self) -> usize {
        assert_eq!(self.counts.fetch_add(1, Ordering::SeqCst), 0);
        2
    }
    fn items_for_shard(&self, shard: i64) -> Vec<SeedItem> {
        assert_eq!(
            self.queries[shard as usize].fetch_add(1, Ordering::SeqCst),
            0
        );
        if shard == 0 {
            vec![SeedItem {
                seed: -9,
                offset: 0,
                candidate: good_candidate(" exact captured question "),
            }]
        } else {
            vec![]
        }
    }
}
#[tokio::test]
async fn source_is_captured_once_and_exact_captured_values_are_executed() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let source = ChangingSource {
        counts: AtomicUsize::new(0),
        queries: [AtomicUsize::new(0), AtomicUsize::new(0)],
    };
    let prepared = engine.prepare(&source).unwrap();
    let expected = prepared.manifest().clone();
    let report = engine
        .run_prepared(
            "captured",
            prepared,
            gw_storage::RunMode::CreateOrResume,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(report.completed);
    let calls = teacher.seen_requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].seed, Some(-9));
    assert_eq!(
        calls[0].messages[0].content,
        gw_schema::Content::Text(" exact captured question ".into())
    );
    let saved: String = sqlx::query_scalar("SELECT config_json FROM runs WHERE run_id='captured'")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<gw_schema::RunManifest>(&saved).unwrap(),
        expected
    );
    assert_eq!(source.counts.load(Ordering::SeqCst), 1);
    for count in source.queries {
        assert_eq!(count.load(Ordering::SeqCst), 1)
    }
}
async fn engine(area: AreaConfig) -> Engine {
    let store = Store::open_in_memory().await.unwrap();
    Engine::new(
        clients(
            store,
            Arc::new(ExplodingTeacher),
            Arc::new(ScriptedJudge::new(vec![])),
            EventSink::disconnected(),
        ),
        area,
        1,
    )
}
#[tokio::test]
async fn effective_requests_and_admission_fields_are_pinned_while_clamps_are_equivalent() {
    let base = area_k1(three_judges(), lenient_thresholds())
        .with_admission_intent(AdmissionIntent::ReviewOnly);
    let before = engine(base.clone())
        .await
        .prepare(&one_item_source())
        .unwrap()
        .manifest()
        .clone();
    type Change = fn(&mut AreaConfig);
    let changes: Vec<Change> = vec![
        |a| a.teacher_slug.push('x'),
        |a| a.training_area.push('x'),
        |a| a.max_tokens += 1,
        |a| a.teacher_reasoning_max_tokens = Some(4000),
        |a| a.k = 2,
        |a| a.cot_required = false,
        |a| a.rubric.push(' '),
        |a| a.correlation_rho = 0.5,
        |a| a.thresholds.accept_threshold = 0.9,
        |a| a.thresholds.reject_below = 0.4,
        |a| a.thresholds.min_n_eff = 1.1,
        |a| a.thresholds.min_n_eff_ratio = 0.2,
        |a| a.admission_intent = AdmissionIntent::Automatic,
        |a| a.judges.reverse(),
        |a| a.judges[0].slug.push('x'),
        |a| a.judges[0].family.push('x'),
        |a| a.judges[0].rubric_id.as_mut().unwrap().push('x'),
        |a| a.judges[0].sampling.temperature = 0.3,
        |a| a.judges[0].sampling.top_p = Some(0.7),
        |a| a.judges[0].sampling.seed = Some(-7),
        |a| a.judges[0].reasoning = None,
        |a| {
            a.judges[0].reasoning = Some(gw_judge::JudgeReasoning::Effort(
                gw_schema::ReasoningEffort::High,
            ))
        },
        |a| a.judges[0].max_tokens = 9999,
    ];
    for change in changes {
        let mut changed = base.clone();
        change(&mut changed);
        let actual = engine(changed)
            .await
            .prepare(&one_item_source())
            .unwrap()
            .manifest()
            .clone();
        assert_ne!(actual, before);
    }
    let mut zero = base.clone();
    zero.k = 0;
    assert_eq!(
        engine(zero)
            .await
            .prepare(&one_item_source())
            .unwrap()
            .manifest(),
        &before
    );
    let mut zero_judge = base.clone();
    zero_judge.judges[0].max_tokens = 0;
    let mut one_judge = zero_judge.clone();
    one_judge.judges[0].max_tokens = 1;
    assert_eq!(
        engine(zero_judge)
            .await
            .prepare(&one_item_source())
            .unwrap()
            .manifest(),
        engine(one_judge)
            .await
            .prepare(&one_item_source())
            .unwrap()
            .manifest()
    );
    assert_eq!(
        before.unattested_deployment,
        gw_schema::UnattestedDeployment::default()
    );
}
#[tokio::test]
async fn invalid_actual_floats_are_rejected_before_json_can_turn_them_into_null() {
    type Change = fn(&mut AreaConfig, f64);
    let fields: Vec<Change> = vec![
        |a, x| a.correlation_rho = x,
        |a, x| a.thresholds.accept_threshold = x,
        |a, x| a.thresholds.reject_below = x,
        |a, x| a.thresholds.min_n_eff = x,
        |a, x| a.thresholds.min_n_eff_ratio = x,
        |a, x| a.judges[0].sampling.temperature = x,
        |a, x| a.judges[0].sampling.top_p = Some(x),
    ];
    for field in fields {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut area = area_k1(one_judge(), lenient_thresholds());
            field(&mut area, bad);
            assert!(engine(area).await.prepare(&one_item_source()).is_err());
        }
    }
}

#[tokio::test]
async fn invalid_plan_never_registers_a_launch_or_run() {
    for items in [
        vec![item(0, 1)],
        vec![item(0, u64::MAX)],
        vec![item(0, 0), item(0, 1)],
        vec![item(0, 0), item(1, 0)],
    ] {
        let store = Store::open_in_memory().await.unwrap();
        let engine = Engine::new(
            clients(
                store.clone(),
                Arc::new(ExplodingTeacher),
                Arc::new(ScriptedJudge::new(vec![])),
                EventSink::disconnected(),
            ),
            area_k1(one_judge(), lenient_thresholds()),
            1,
        );
        assert!(
            engine
                .run(
                    "invalid",
                    &Plan {
                        count: 1,
                        shards: vec![items]
                    },
                    CancellationToken::new()
                )
                .await
                .is_err()
        );
        assert!(store.run_status("invalid").await.unwrap().is_none());
        assert!(store.model_launches("invalid").await.unwrap().is_empty());
    }
}

#[test]
fn task_policies_and_exact_required_test_ids_participate_in_input_identity() {
    let base = numeric_candidate("compute", "42");
    let hash = digest(base.clone());
    for policy in [
        gw_schema::VerificationPolicy::Absent,
        gw_schema::VerificationPolicy::Authoritative,
    ] {
        let mut changed = base.clone();
        changed.contract.answer_policy = Some(policy);
        assert_ne!(hash, digest(changed));
    }
    let mut execution = base.clone();
    execution.contract.execution_policy = Some(gw_schema::VerificationPolicy::Advisory);
    assert_ne!(hash, digest(execution.clone()));
    execution.contract.required_tests = vec!["test::node".into()];
    let original = digest(execution.clone());
    execution.contract.required_tests[0] = " test::node ".into();
    assert_ne!(original, digest(execution.clone()));
    execution.contract.execution_policy = Some(gw_schema::VerificationPolicy::Authoritative);
    assert_ne!(original, digest(execution));
}
