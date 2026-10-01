//! Private supplied snapshots and profiles: no real artifact/deployment qualification.
// Individual integration targets use different subsets of the shared fixtures.
#![allow(dead_code)]
use gw_providers::serving_profile::*;
use gw_schema::*;
use serde_json::json;

pub fn digest(byte: char) -> ContentDigest {
    ContentDigest {
        algorithm: DigestAlgorithm::Sha256,
        hex: byte.to_string().repeat(64),
    }
}
pub fn declaration(name: &str) -> SemanticDeclaration {
    SemanticDeclaration::new(name, "1", json!({}))
}
pub fn artifact() -> PinnedModelArtifact {
    let source = PinnedModelSource {
        reference: ModelReference::new("urn:fixture:model").unwrap(),
        revision: "immutable-1".into(),
    };
    let file = |path: &str, purpose, byte| ModelArtifactFile {
        path: path.into(),
        purpose,
        content: digest(byte),
    };
    let tokenizer = ModelComponentReference {
        source: source.clone(),
        file: file("tokenizer.json", ModelFilePurpose::Tokenizer, 'b'),
    };
    let template = ModelComponentReference {
        source: source.clone(),
        file: file("chat.jinja", ModelFilePurpose::ChatTemplate, 'c'),
    };
    PinnedModelArtifact {
        version: 1,
        label: "Private synthetic fixture".into(),
        source,
        files: vec![
            file("weights.bin", ModelFilePurpose::Weights, 'a'),
            tokenizer.file.clone(),
            template.file.clone(),
        ],
        lineage: ModelArtifactLineage::Base {},
        tokenizer: Declaration::Declared(tokenizer),
        chat_template: Declaration::Declared(Some(template)),
    }
}
pub fn profile(dialect: ServingDialect) -> ServingProfile {
    let mut capabilities = vec![
        ProfileControl::Temperature,
        ProfileControl::TopP,
        ProfileControl::Seed,
        ProfileControl::MaxOutputTokens,
        ProfileControl::ReasoningEffort,
        ProfileControl::JsonSchema,
        ProfileControl::Logprobs,
        ProfileControl::Usage,
        ProfileControl::ReasoningHistory,
    ];
    let (operations, reasoning_efforts, routing) = match dialect {
        ServingDialect::OpenRouterV1 => {
            capabilities.extend([
                ProfileControl::ReasoningBudget,
                ProfileControl::StructuredReasoningHistory,
            ]);
            (
                vec![ModelOperation::ChatCompletion],
                vec!["xhigh".into(), "low".into()],
                Some(StrictProviderRouting {
                    only: vec!["replica-b".into(), "replica-a".into()],
                    data_collection: DataCollection::Deny,
                }),
            )
        }
        ServingDialect::VllmV1 => {
            capabilities.extend([
                ProfileControl::EnableThinking,
                ProfileControl::PreserveThinking,
                ProfileControl::ClearThinking,
            ]);
            (
                vec![
                    ModelOperation::ChatCompletion,
                    ModelOperation::Completion,
                    ModelOperation::Embedding,
                ],
                vec!["max".into(), "low".into()],
                None,
            )
        }
    };
    ServingProfile {
        version: 1,
        endpoint: "https://serve.example.test/v1/".into(),
        authentication: ProfileAuthentication::None {},
        behavior: ProfileBehavior {
            version: 1,
            dialect,
            operations,
            capabilities,
            reasoning_efforts,
        },
        routing,
        operations: ProfileOperations::default(),
    }
}
pub fn semantics(profile: &ServingProfile, operation: ModelOperation) -> ModelExecutionSemantics {
    let artifact = artifact();
    ModelExecutionSemantics {
        version: 1,
        alias: "private-fixture".into(),
        operation,
        adapter_behavior: profile.behavior.adapter_behavior(operation).unwrap(),
        serving_profile: Declaration::Declared(profile.behavior.declaration().unwrap()),
        artifact: Declaration::Declared(artifact.identity().unwrap()),
        additional_artifacts: Declaration::Declared(vec![]),
        tokenizer: artifact.tokenizer,
        chat_template: if operation == ModelOperation::ChatCompletion {
            artifact.chat_template
        } else {
            Declaration::Declared(None)
        },
        runtime: Declaration::Declared(declaration("fixture-runtime")),
        parser: Declaration::Declared(declaration("fixture-reasoning-parser")),
        configuration: Declaration::Declared(digest('d')),
    }
}
pub fn message() -> Message {
    Message {
        role: Role::User,
        content: Content::Text("hello".into()),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}
pub fn chat() -> ProfileRequest {
    ProfileRequest {
        input: ProfileInput::Chat(vec![message()]),
        controls: vec![],
    }
}
pub fn required(value: ControlValue) -> RequestedControl {
    RequestedControl {
        value,
        required: true,
    }
}
pub fn optional(value: ControlValue) -> RequestedControl {
    RequestedControl {
        value,
        required: false,
    }
}
pub fn prepared(profile: &ServingProfile) -> PreparedProfileRequest {
    prepare_request(
        profile,
        &semantics(profile, ModelOperation::ChatCompletion),
        &chat(),
    )
    .unwrap()
}

pub fn deployment(
    prepared: &PreparedProfileRequest,
    instance: &str,
    incarnation: &str,
) -> ModelDeploymentEvidence {
    let artifact = match &prepared.semantics().artifact {
        Declaration::Declared(value) => value.clone(),
        _ => panic!("complete fixture"),
    };
    let mut loaded = vec![artifact];
    if let Declaration::Declared(extra) = &prepared.semantics().additional_artifacts {
        loaded.extend(extra.iter().cloned());
    }
    ModelDeploymentEvidence {
        version: 1,
        method: SemanticDeclaration::new(
            "ghostwriter/loaded-generation-report",
            "1",
            json!({"scope":"loaded_generation"}),
        ),
        issuer: ModelReference::new("urn:fixture:unauthenticated-issuer").unwrap(),
        verifier: Declaration::Declared(declaration("fixture-unqualified-verifier-claim")),
        raw_evidence: PinnedModelDocument {
            source: PinnedModelSource {
                reference: ModelReference::new(format!(
                    "urn:fixture:report:{instance}:{incarnation}"
                ))
                .unwrap(),
                revision: "1".into(),
            },
            content: digest('e'),
        },
        endpoint: prepared.endpoint().clone(),
        instance: instance.into(),
        incarnation: incarnation.into(),
        claimed_loaded_artifacts: Declaration::Declared(loaded),
        effective: prepared.semantics().clone(),
        validity: ModelEvidenceValidity {
            not_before_unix_ms: 100,
            expires_at_unix_ms: Some(1000),
            revocation: Declaration::Declared(ModelRevocationEvidence {
                document: PinnedModelDocument {
                    source: PinnedModelSource {
                        reference: ModelReference::new("urn:fixture:revocations").unwrap(),
                        revision: "1".into(),
                    },
                    content: digest('f'),
                },
                observed_at_unix_ms: 200,
            }),
        },
    }
}
pub fn gateway_request(prepared: &PreparedProfileRequest, attempt: &str) -> GatewayRequest {
    GatewayRequest::from_prepared(
        prepared,
        format!("correlation-{attempt}"),
        ModelAttemptReference {
            run_id: "run-fixture".into(),
            launch_id: "launch-fixture".into(),
            attempt_id: attempt.into(),
            observation_sequence: None,
        },
        None,
    )
    .unwrap()
}
pub fn gateway_response(
    request: &GatewayRequest,
    deployment: &ModelDeploymentEvidence,
) -> GatewayResponseEvidence {
    GatewayResponseEvidence {
        version: 1,
        target: Declaration::Declared(request.target.clone()),
        deployment_evidence: Declaration::Declared(deployment.identity().unwrap()),
        endpoint: request.endpoint.clone(),
        request_body_digest: request.request_body_digest.clone(),
        correlation_id: request.correlation_id.clone(),
        attempt: request.attempt.clone(),
    }
}
pub fn supplied_validity(deployment: &ModelDeploymentEvidence) -> SuppliedValiditySnapshot {
    let revocation = match &deployment.validity.revocation {
        Declaration::Declared(value) => value.clone(),
        _ => panic!("known fixture"),
    };
    SuppliedValiditySnapshot {
        version: 1,
        deployment_evidence: deployment.identity().unwrap(),
        revocation,
        revoked: Declaration::Declared(false),
        valid_until_unix_ms: 900,
    }
}

#[derive(Clone)]
pub struct GatewayFixture {
    pub prepared: PreparedProfileRequest,
    pub request: GatewayRequest,
    pub response: GatewayResponseEvidence,
    pub deployment: ModelDeploymentEvidence,
    pub artifacts: Vec<PinnedModelArtifact>,
    pub validity: SuppliedValiditySnapshot,
}
impl GatewayFixture {
    pub fn new() -> Self {
        Self::with_artifact(artifact())
    }
    pub fn with_artifact(artifact: PinnedModelArtifact) -> Self {
        let profile = profile(ServingDialect::VllmV1);
        let mut semantics = semantics(&profile, ModelOperation::ChatCompletion);
        semantics.artifact = Declaration::Declared(artifact.identity().unwrap());
        let prepared = prepare_request(&profile, &semantics, &chat()).unwrap();
        let request = gateway_request(&prepared, "attempt-1");
        let deployment = deployment(&prepared, "replica-a", "generation-1");
        let response = gateway_response(&request, &deployment);
        let validity = supplied_validity(&deployment);
        Self {
            prepared,
            request,
            response,
            deployment,
            artifacts: vec![artifact],
            validity,
        }
    }
    pub fn report(&self) -> GatewayConsistency {
        check_gateway_consistency(
            &self.prepared,
            &self.request,
            &self.response,
            Some(&self.deployment),
            &self.artifacts,
            Some(&self.validity),
            500,
        )
    }
    pub fn refresh_replica_binding(&mut self) {
        let identity = self.deployment.identity().unwrap();
        self.response.deployment_evidence = Declaration::Declared(identity.clone());
        self.validity.deployment_evidence = identity;
    }
}
pub fn mismatch(axis: &ConsistencyAxis, reason: ConsistencyReason) {
    assert_eq!(axis.state(), ConsistencyState::Mismatch, "{axis:?}");
    assert!(axis.reasons().contains(&reason), "{axis:?}");
}
pub fn unknown(axis: &ConsistencyAxis, reason: ConsistencyReason) {
    assert_eq!(axis.state(), ConsistencyState::Unknown, "{axis:?}");
    assert!(axis.reasons().contains(&reason), "{axis:?}");
}
