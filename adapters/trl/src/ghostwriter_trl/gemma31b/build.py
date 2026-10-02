"""Build auditable examples and a manifest without reconstructing Rust's logical hash."""
from hashlib import sha256
import json
import platform

from .. import __version__
from ..artifact import ContractError, VerifiedSnapshot, strict_json
from .projection import prepare_target, project_messages
from .policy import PROFILE, controls
from ..screening import consumer_screening
from .tokenizer import PACKAGE, check_dependencies, tokenizer_manifest, tokenizer_policy, validate_tokenizer


def identity(value) -> str:
    """Adapter identities use an explicitly separate SHA256 JSON domain from Rust artifacts."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity() -> str:
    """Bind all shared preparation and recursive profile source/policies from the installed package.

    The separate optimizer/checkpoint package is not consumed by preparation or dataloader replay.
    """
    from ..build import source_identity as shared_identity
    return identity({"shared_preparation": shared_identity(), "serial_tools": {
        str(path.relative_to(PACKAGE)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(PACKAGE.rglob("*")) if path.is_file() and path.suffix in {".py", ".json"}}})


def build(snapshot: VerifiedSnapshot, tokenizer, *, cot: str, turns: str, max_length: int,
          profile: str = PROFILE, enable_thinking: bool, preserve_thinking: bool) -> tuple[list[dict], dict]:
    """Explicitly expand assistant prefixes; rejected targets remain counted and identified."""
    if type(snapshot) is not VerifiedSnapshot:
        raise ContractError("source must be an actual verified-origin snapshot")
    if cot not in {"supervised", "masked", "stripped"} or turns not in {"final_turn_only", "all_assistant"}:
        raise ContractError("explicit supported reasoning and assistant-turn policies are required")
    if type(max_length) is not int or max_length < 1:
        raise ContractError("max_length must be a positive integer")
    verification_report = snapshot.report
    source_artifact = verification_report["artifact"]
    screening = consumer_screening(source_artifact, cot, turns)
    dependencies = check_dependencies(profile)
    validate_tokenizer(tokenizer, profile)
    settings = controls(profile, enable_thinking, preserve_thinking)
    pinned_tokenizer = tokenizer_manifest(profile)
    recipe = {
        "version": 2, "adapter_version": __version__, "adapter_source_sha256": source_identity(),
        "preparation_profile": {"name": profile, "controls": settings},
        "dependencies": dependencies, "tokenizer": pinned_tokenizer, "tokenizer_policy": tokenizer_policy(profile),
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(), "system": platform.system(), "machine": platform.machine()},
        "tokenizer_target": {"repository": pinned_tokenizer["repository"], "revision": pinned_tokenizer["revision"]},
        "cot_policy": cot, "multi_turn_loss": turns,
        "layout": "assistant_prefix_v1" if turns == "all_assistant" else "full_conversation_final_v1",
        "max_length": max_length, "offset_unit": "python_unicode_codepoint",
        "labels": "unshifted_causal_lm", "add_special_tokens": False,
        "truncation": False,
    }
    if screening is not None:
        recipe["screening"] = {key: screening["plan"][key] for key in (
            "plan_id", "policy_id", "screening_input_id", "protected_input_id", "grouping_id",
        )}
        recipe["screening"]["population_id"] = screening["population_id"]
    recipe_id = identity(recipe)
    rows = snapshot.rows()
    if len(rows) != source_artifact["manifest"]["n_admitted"]:
        raise ContractError("PyArrow row count differs from verified artifact")
    components = {} if screening is None else {
        member["record"]["record_id"]: member for member in screening["members"]
    }
    if screening is not None and (
            len(components) != len(screening["members"])
            or set(components) != {row["record_id"] for row in rows}):
        raise ContractError("screened row membership mismatch")
    projections = {} if screening is None else {
        (binding["record"]["run_id"], binding["record"]["record_id"]): binding["export_projection_id"]
        for binding in screening["plan"]["population"]
    }
    examples, rejections = [], []
    candidate_targets, accepted_records = 0, set()
    for row in rows:
        source = {key: row[key] for key in ("record_id", "record_hash", "prompt_hash", "messages_json")}
        source["task_json"] = row.get("task_json")
        if "tools_json" in row:
            source["tools_json"] = row["tools_json"]
        origin = None
        if "origin_json" in row:
            from ..origin import row_origin
            origin = row_origin(row)
            source["origin_json"] = row["origin_json"]
        source["artifact_id"] = source_artifact["artifact_id"]
        # The record itself is the grouping boundary available here. Split membership stays unknown.
        source["group_kind"] = "source_record"
        source["group_id"] = identity(["ghostwriter.source-group.v1", source_artifact["artifact_id"], row["record_id"]])
        if source["task_json"] is not None:
            declarations = strict_json(source["task_json"])["provenance"]
            source["declared_task"] = {key: declarations[key] for key in ("identity", "group", "split", "rights")}
            source["group_kind"] = "declared_task_group"
            source["group_id"] = identity(["ghostwriter.declared-task-group.v1", declarations["group"]])
            if declarations["split"]["role"] != "train":
                rejections.append({"record_id": row["record_id"], "target_index": None, "reason": "declared held-out task role is excluded from SFT training preparation"})
                continue
        if origin is not None and origin["kind"] == "reviewed_reference":
            source["group_kind"] = "reviewed_reference_component"
            source["group_id"] = identity(["ghostwriter.reference-component.v1", origin["catalogue_id"], origin["component"]])
        if screening is not None:
            member = components[row["record_id"]]
            source["group_kind"] = "screened_connected_component"
            source["group_id"] = member["component_id"]
            source["screening"] = {**recipe["screening"], "record": member["record"],
                "component_id": member["component_id"],
                "export_projection_id": projections[(member["record"]["run_id"], member["record"]["record_id"])]}
        try:
            if "tools_json" not in row:
                raise ContractError("serial tool preparation requires a complete v5 source")
            definitions = strict_json(row["tools_json"]) if row["tools_json"] is not None else []
            messages = project_messages(strict_json(row["messages_json"]), definitions, cot)
        except ContractError as error:
            rejections.append({"record_id": row["record_id"], "target_index": None, "reason": str(error)})
            continue
        targets = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if turns == "final_turn_only":
            targets = targets[-1:]
        candidate_targets += len(targets)
        for target in targets:
            try:
                example = prepare_target(messages[:target + 1], tokenizer, cot, max_length,
                                         tools=definitions, settings=settings)
            except ContractError as error:
                rejections.append({"record_id": row["record_id"], "target_index": target, "reason": str(error)})
                continue
            example["source"] = source
            example["target_index"] = target
            example["example_id"] = identity(["ghostwriter.sft-example.v1", recipe_id, source, target, example["input_ids"], example["labels"]])
            example["supervised_tokens"] = sum(label != -100 for label in example["labels"])
            example["context_tokens"] = len(example["labels"]) - example["supervised_tokens"]
            example["effective_shifted_supervised_tokens"] = sum(label != -100 for label in example["labels"][1:])
            examples.append(example)
            accepted_records.add(row["record_id"])
    manifest = {
        "build_manifest_version": 1, "recipe_id": recipe_id, "recipe": recipe,
        "source_record_count": len(rows), "accepted_source_record_count": len(accepted_records),
        "candidate_target_count": candidate_targets, "expanded_example_count": len(examples),
        "rejected_item_count": len(rejections), "rejections": rejections,
        "rejected_record_count": sum(r["target_index"] is None for r in rejections),
        "rejected_target_count": sum(r["target_index"] is not None for r in rejections),
        "source_records_with_no_examples": len(rows) - len(accepted_records),
        "example_ids": [e["example_id"] for e in examples],
        "supervised_token_count": sum(e["supervised_tokens"] for e in examples),
        "context_token_count": sum(e["context_tokens"] for e in examples),
        "effective_shifted_supervised_token_count": sum(e["effective_shifted_supervised_tokens"] for e in examples),
        "effective_shifted_call_token_count": sum(len(e["shifted_call_token_indices"]) for e in examples),
        "effective_shifted_answer_token_count": sum(len(e["shifted_answer_token_indices"]) for e in examples),
        "qualification_limits": {
            "lifecycle_eligibility": "not_reconstructed_by_artifact_verification",
            "rights_and_execution_lineage": "unknown", "student_parent_revision": None,
            "grouped_split_qualification": "unknown", "contamination_screening": "unknown",
            "heldout_training_benefit": "unknown",
            "semantic_screening": "not_run", "effective_prompt_separation": "unknown",
            "student_weights": "unbound", "execution_lineage": "unbound", "decision_lineage": "unbound",
        },
    }
    if screening is not None:
        manifest["qualification_limits"].update({
            "grouped_split_qualification": "declared_train_components_source_screened",
            "contamination_screening": screening["plan"]["lexical_status"],
            "screening_population": screening["population_check"],
            "screening_lexical_scope": screening["plan"]["lexical_scope"],
            "semantic_screening": "not_run", "effective_prompt_separation": "unknown",
        })
    return examples, manifest
