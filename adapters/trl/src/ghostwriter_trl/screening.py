"""Strict consumer policy for Rust-verified source screening; no screening or hashing algorithm."""
from .artifact import ContractError


def _shape(value, keys, what):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ContractError(f"unexpected {what} fields")


def validate_screening_shape(artifact):
    """Check supported version-specific witness shapes after Rust verifies the same bytes."""
    witness = artifact["screening"]
    _shape(witness, "version validation population_check population_id layout plan members", "screening witness")
    if (type(witness["version"]) is not int or witness["version"] != 2
            or witness["validation"] != "planner_rerun_v2"
            or witness["population_check"] != "transaction_checked"):
        raise ContractError("unsupported screened publication qualification")
    plan = witness["plan"]
    _shape(plan, "version counts declaration population required_fields screening_input_id policy_id protected_inputs protected_input_id groups grouping_id edges protected_matches lexical_status semantic_status population_check lexical_scope effective_prompt_separation incomplete eligible_output exclusions strata previous plan_id", "screening plan")
    if (type(plan["version"]) is not int or plan["version"] != 2
            or plan["lexical_status"] not in {"complete_no_match", "match_quarantined"}
            or plan["semantic_status"] != "not_run" or plan["population_check"] != "supplied_files_only"
            or plan["lexical_scope"] != "canonical_source_and_pinned_export_policy"
            or plan["effective_prompt_separation"] != "unknown" or plan["incomplete"] != []):
        raise ContractError("unsupported or incomplete source screening report")
    declaration = plan["declaration"]
    _shape(declaration, "version runs output policy expected_tasks siblings", "screening declaration")
    if type(declaration["version"]) is not int or declaration["version"] != 1:
        raise ContractError("unsupported screening declaration version")
    policy = declaration["policy"]
    _shape(policy, "recipe ngram min_overlap_tokens jaccard_threshold limits target cot_policy multi_turn_loss additional_protected_sets required_languages", "screening policy")
    if policy["recipe"] != "lexical-screen-v1":
        raise ContractError("unsupported source screening recipe")
    if not isinstance(witness["members"], list):
        raise ContractError("invalid screened membership")
    for member in witness["members"]:
        _shape(member, "record component_id", "screened member")
        _shape(member["record"], "run_id record_id", "screened record coordinate")
    if not isinstance(plan["population"], list):
        raise ContractError("invalid screened population")
    for binding in plan["population"]:
        _shape(binding, "record record_hash export_projection_id screening_input_id", "screening input binding")
    return witness


def consumer_screening(artifact, cot, turns):
    """Reject policy mismatch even on zero-row inputs and before decoding any example."""
    if artifact["metadata_version"] == 1:
        return None
    witness = validate_screening_shape(artifact)
    manifest = artifact["manifest"]
    policy = witness["plan"]["declaration"]["policy"]
    layout = "assistant_prefix_v1" if turns == "all_assistant" else "full_conversation_final_v1"
    if (manifest["column_schema_version"] not in {"reviewed_tasks", "record_origins", "tool_definitions"}
            or manifest["target"] != "open_ai_messages" or policy["target"] != "open_ai_messages"
            or policy["cot_policy"] != cot or manifest["cot_policy"] != cot
            or policy["multi_turn_loss"] != turns or manifest["multi_turn_loss"] != turns
            or witness["layout"] != layout):
        raise ContractError("screened artifact target, reasoning, turn policy or layout does not match consumer")
    return witness
