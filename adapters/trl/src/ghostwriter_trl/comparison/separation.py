"""Bind real registered Train membership and check the actual generation-rendered token inputs."""
import blake3

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes
from ..build import identity
from ..lora.config import FIXTURE
from ..profiles import GEMMA
from ..projection import project_messages
from .protocol import render_prompt


def check_separation(prepared, population, tokenizer, settings, models):
    """Check all supplied source groups/components and exact rendered inputs, with honest limits."""
    heldout_groups = {identity(member["provenance"]["group"]) for member in population["members"]}
    heldout_components = {identity(member["component"]) for member in population["members"]}
    heldout_tasks = {member["provenance"]["identity"]["digest"] for member in population["members"]}
    accepted_train = {member["member_id"]: member for member in population["training_members"]}
    expected_train = set(accepted_train)
    train_members = set()
    all_references = True
    training_prompts = set()

    def system(messages):
        if settings["system_prompt"]:
            if messages and messages[0]["role"] == "system":
                messages = messages[1:]
            return [{"role": "system", "content": settings["system_prompt"]}] + messages
        return messages

    for example in prepared.examples:
        source = example["source"]
        task = source.get("declared_task")
        if task is not None and (task["split"]["role"] != "train"
                or identity(task["group"]) in heldout_groups or task["identity"]["digest"] in heldout_tasks):
            raise ContractError("prepared training task overlaps the captured held-out task or family")
        origin = strict_json(source["origin_json"]) if source.get("origin_json") else None
        if origin is None or origin["kind"] != "reviewed_reference":
            all_references = False
        else:
            if (any(origin[key] != population[key] for key in ("catalogue_id", "registration_id", "batch_id"))
                    or origin["member_id"] not in expected_train or identity(origin["component"]) in heldout_components):
                raise ContractError("prepared reference differs from accepted Train membership or crosses a component")
            accepted = accepted_train[origin["member_id"]]
            content_id = blake3.blake3(_json_bytes([source.get(key) for key in
                ("record_id", "messages_json", "task_json", "origin_json")]),
                derive_key_context="ghostwriter.coding-training-content.v1").hexdigest()
            if (content_id != accepted["content_id"] or source["record_id"] != accepted["record_id"]
                    or task is None or task["identity"] != accepted["task_identity"] or task["group"] != accepted["group"]
                    or origin["component"] != accepted["component"] or origin["suite_id"] != accepted["suite_id"]):
                raise ContractError("prepared reference content differs from the exact accepted Train record")
            train_members.add(origin["member_id"])
        messages = project_messages(strict_json(source["messages_json"]), tokenizer, "stripped", profile=GEMMA)
        context = [{k: v for k, v in message.items() if k != "reasoning_content" or message["role"] == "assistant"}
                   for message in messages[:example["target_index"]]]
        rendered = render_prompt(tokenizer, system(context), settings["max_prompt_tokens"])
        training_prompts.add(tuple(rendered["input_ids"]))
    heldout_prompts, unrendered = [], []
    for member in population["members"]:
        try:
            prompt = render_prompt(tokenizer, system([{"role": "user", "content": member["prompt"]}]),
                                   settings["max_prompt_tokens"])
        except ContractError:
            unrendered.append(member["member_id"])
        else:
            heldout_prompts.append(tuple(prompt["input_ids"]))
    if len(set(heldout_prompts)) != len(heldout_prompts) or training_prompts.intersection(heldout_prompts):
        raise ContractError("effective generation prompt IDs collide within held-out or training populations")
    complete_references = all_references and train_members == expected_train
    if not complete_references and models["source_authorization"] != FIXTURE:
        raise ContractError("real comparison requires all 64 accepted reference Train members in the prepared build")
    return {"training_population": "registered_reference_train" if complete_references else "owned_software_fixture",
            "training_member_ids": list(accepted_train) if complete_references else [],
            "source_screening": "declared_source_screened" if "screening" in prepared.manifest["recipe"] else "not_run",
            "semantic_screening": "not_run", "effective_prompt_separation": "incomplete_unrenderable_heldout" if unrendered else "generation_recipe_prompt_ids_disjoint",
            "unrendered_member_ids": unrendered}
