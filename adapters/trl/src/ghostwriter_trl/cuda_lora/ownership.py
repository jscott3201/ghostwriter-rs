"""Model-free successful completions own captured bytes until explicit close or consumption."""
from contextlib import contextmanager
from pathlib import Path
import json
import tempfile
from weakref import WeakKeyDictionary, finalize

from ..artifact import ContractError, strict_json
from ..lora.safe_tensors import regular
from ..training.publication import PublishedCheckpointError
from .lifecycle import clear_tracebacks

_LIVE = WeakKeyDictionary()
_CLEANUPS = WeakKeyDictionary()


class _Capture:
    def __init__(self, path, limit):
        self.workspace = tempfile.TemporaryDirectory(prefix="gw-cuda-completion-")
        self.path = Path(self.workspace.name) / "completion.gwckpt"
        try:
            with regular(path) as source, self.path.open("xb") as destination:
                size = 0
                while data := source.read(1024**2):
                    size += len(data)
                    if size > limit:
                        raise ContractError("CUDA completion exceeds capture bound")
                    destination.write(data)
            self.path.chmod(0o400)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.workspace.cleanup()


class ObservedCompletion:
    """Only a successful live producer can mint this single-consumption capability."""
    __slots__ = ("__receipt", "__weakref__")

    def __new__(cls, *args, **kwargs):
        raise TypeError("CUDA completions are created only by successful owned training")

    def __setattr__(self, name, value):
        raise AttributeError("CUDA completion is immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("CUDA completion cannot be subclassed")

    @property
    def completion_id(self):
        return self.__receipt[0]

    @property
    def observed(self):
        return strict_json(self.__receipt[1])

    def close(self):
        """Revoke live eligibility and release the owned immutable capture."""
        _release(self)

    def __enter__(self):
        if self not in _LIVE:
            raise ContractError("CUDA completion is closed or consumed")
        return self

    def __exit__(self, kind, error, traceback):
        _release(self, error)


def _release(completion, consumer_error=None):
    _LIVE.pop(completion, None)
    state = _CLEANUPS.get(completion)
    if state is None:
        return
    capture, finalizer = state
    try:
        capture.close()
    except BaseException as error:
        failures = [] if consumer_error is None else [("consumer", consumer_error)]
        failures.append(("capture_cleanup", error))
        clear_tracebacks(value for _, value in failures)
        receipt = completion._ObservedCompletion__receipt
        prepared_id, output, synced = receipt[2]
        raise PublishedCheckpointError(receipt[0], prepared_id, output, synced, failures) from error
    finalizer.detach()
    _CLEANUPS.pop(completion, None)


def _mint(capture, completion_id, observed, source_id, publication):
    result = object.__new__(ObservedCompletion)
    if not publication.published or publication.completion_id != completion_id:
        raise ContractError("live completion requires its actual publication outcome")
    receipt = (completion_id, json.dumps(observed, allow_nan=False),
               (publication.prepared_id, str(publication.output), publication.directory_synced))
    object.__setattr__(result, "_ObservedCompletion__receipt", receipt)
    _CLEANUPS[result] = (capture, finalize(result, capture.close))
    _LIVE[result] = (capture, receipt, source_id)
    return result


@contextmanager
def consume(completion):
    """Transfer the actual owned capture once; saved reports and public paths confer no authority."""
    from .producer import training_source_identity
    if type(completion) is not ObservedCompletion or completion not in _LIVE:
        raise ContractError("requires an unconsumed live owned CUDA completion")
    capture, receipt, source = _LIVE.pop(completion)
    consumer_error = None
    try:
        if getattr(completion, "_ObservedCompletion__receipt", None) is not receipt or source != training_source_identity():
            raise ContractError("CUDA completion or its producer source changed")
        yield capture.path, receipt[0], strict_json(receipt[1])
    except BaseException as error:
        consumer_error = error
        raise
    finally:
        _release(completion, consumer_error)
