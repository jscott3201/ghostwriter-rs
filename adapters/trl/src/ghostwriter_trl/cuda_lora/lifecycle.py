"""Release object-bearing traceback frames without hiding primary or cleanup failures."""
from ..artifact import ContractError


def clear_tracebacks(errors):
    """Detach every chained traceback before retaining or rethrowing a model failure."""
    pending = list(errors)
    seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        error.__traceback__ = None
        pending.extend(cause for cause in (error.__cause__, error.__context__) if cause is not None)


def reload_failure(errors):
    """Retain semantic diagnostics while cleanup independently attempts all resources."""
    clear_tracebacks(error for _, error in errors)
    if len(errors) == 1:
        raise errors[0][1]
    detail = "; ".join(f"{phase}: {type(error).__name__}: {error}" for phase, error in errors)
    raise ContractError(detail) from errors[0][1]
