"""Pinned isolated function adapter; expected values never enter this process.

This wrapper is not a security boundary. The external controller owns isolation,
resource limits, output parsing, expected results, and every pass/fail decision.
"""
import ast
import inspect
import json
import sys
import types

if sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 14):
    raise RuntimeError("unsupported interpreter")


def decode(value):
    tag = value["type"]
    if tag == "null":
        return None
    if tag == "array":
        return [decode(item) for item in value["value"]]
    if tag == "object":
        return {key: decode(item) for key, item in value["value"].items()}
    return value["value"]


def encode(value, depth=0, budget=None):
    if budget is None:
        budget = [2048, 16384]
    budget[0] -= 1
    if depth > 16 or budget[0] < 0:
        raise ValueError("result exceeds depth or node bound")
    kind = type(value)
    if value is None:
        return {"type": "null"}
    if kind is bool:
        return {"type": "boolean", "value": value}
    if kind is int and -(2**63) <= value < 2**63:
        return {"type": "integer", "value": value}
    if kind is str:
        budget[1] -= len(value.encode("utf-8"))
        if budget[1] < 0:
            raise ValueError("result exceeds text bound")
        return {"type": "string", "value": value}
    if kind is list:
        return {"type": "array", "value": [encode(v, depth + 1, budget) for v in value]}
    if kind is dict and all(type(key) is str for key in value):
        budget[1] -= sum(len(key.encode("utf-8")) for key in value)
        if budget[1] < 0:
            raise ValueError("result exceeds text bound")
        return {
            "type": "object",
            "value": {key: encode(item, depth + 1, budget) for key, item in value.items()},
        }
    raise TypeError("unsupported exact result type")


request = json.loads(sys.stdin.buffer.read(1048577))
source = request["code"]
function = request["function"]
tree = ast.parse(source, filename="candidate.py", mode="exec")
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == function["entry_point"]]
if len(definitions) != 1:
    raise TypeError("module must define exactly one declared synchronous entry point")
arguments = definitions[0].args
if (arguments.posonlyargs or arguments.vararg or arguments.kwarg or arguments.kwonlyargs
        or arguments.defaults or arguments.kw_defaults
        or [item.arg for item in arguments.args] != function["parameters"]):
    raise TypeError("entry point signature differs from the declared positional signature")
namespace = {"__name__": "candidate", "__file__": "candidate.py"}
exec(compile(tree, "candidate.py", "exec"), namespace)
candidate = namespace[function["entry_point"]]
if type(candidate) is not types.FunctionType or inspect.iscoroutinefunction(candidate):
    raise TypeError("entry point must be a synchronous Python function")
signature = inspect.signature(candidate, follow_wrapped=False)
parameters = list(signature.parameters.values())
if ([parameter.name for parameter in parameters] != function["parameters"]
        or any(parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
               or parameter.default is not inspect.Parameter.empty for parameter in parameters)):
    raise TypeError("runtime entry point signature differs from the declared signature")
result = candidate(*(decode(value) for value in request["arguments"]))
print(json.dumps(encode(result), ensure_ascii=True, separators=(",", ":")))
