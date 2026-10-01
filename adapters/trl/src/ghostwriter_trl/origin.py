"""Strict public origin declarations; local registration authority remains in the source store."""
import blake3

from .artifact import ContractError, strict_json


def row_origin(row: dict) -> dict:
    """Decode exact v4 origin shape and check reference semantics against the same row."""
    origin = strict_json(row["origin_json"])
    if type(origin) is not dict or type(origin.get("version")) is not int or origin["version"] != 1:
        raise ContractError("unsupported record origin")
    if origin.get("kind") == "generated":
        if set(origin) != {"kind", "version"} or row["verdict"] != "admit":
            raise ContractError("invalid generated origin or admission verdict")
        return origin
    fields = {"kind", "version", "catalogue_id", "registration_id", "batch_id", "member_id",
              "reference_code_id", "suite_id", "native_result_id", "authorship", "component", "permitted_use"}
    if origin.get("kind") != "reviewed_reference" or set(origin) != fields:
        raise ContractError("invalid reviewed-reference origin")
    for key in fields - {"kind", "version", "authorship", "component", "permitted_use"}:
        value = origin[key]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError("invalid reference origin digest")
    authorship = origin["authorship"]
    component = origin["component"]
    if (type(authorship) is not dict or set(authorship) != {"author", "reviewer"}
            or any(v not in {"human", "agent"} for v in authorship.values())
            or type(component) is not dict or set(component) != {"namespace", "id"}
            or any(type(v) is not str or not v.strip() for v in component.values())
            or origin["permitted_use"] != "training"):
        raise ContractError("invalid reference authorship, component or permitted use")
    task = strict_json(row["task_json"])
    messages = strict_json(row["messages_json"])
    if (len(messages) != 2 or messages[-1].get("role") != "assistant"
            or set(messages[-1]) != {"role", "content"} or type(messages[-1]["content"]) is not str
            or row["verdict"] is not None or row["judge_aggregate"] is not None or row["reasoning_tokens"] != 0
            or task["provenance"]["split"]["role"] != "train"
            or "training" not in task["provenance"]["rights"]["permitted_uses"]
            or task["verification_contract"]["oracle"].get("suite", {}).get("suite_id") != origin["suite_id"]):
        raise ContractError("reference row contradicts its task or absent generation/judge facts")
    code_id = blake3.blake3(messages[-1]["content"].encode(), derive_key_context="ghostwriter.coding-module.v1").hexdigest()
    if code_id != origin["reference_code_id"]:
        raise ContractError("reference code differs from origin binding")
    return origin
