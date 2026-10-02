"""One owned native process holds the captured private population throughout Python generation."""
from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import tempfile

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes

# Matches native CODING_PAIR_MAX_BYTES; stdout frames include their final newline.
MAX_BYTES = 32 * 1024**2


def _settle(process):
    """Close pending protocol input and await native bounded container/client settlement."""
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.stdin = None
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    # The native controller owns finite per-case, whole-observation and cleanup deadlines.
    # Never kill it while Docker mutations or container cleanup are unsettled.
    while True:
        try:
            process.communicate()
            return
        except KeyboardInterrupt:
            continue


class NativePairBridge:
    """A single live private-oracle capture; public metadata is a detached JSON copy."""
    def __init__(self, process, errors, population):
        self._process = process
        self._errors = errors
        self.population = population
        self._finished = False

    def execute(self, request):
        """Send both complete ordered sides once and receive the settled native paired result."""
        if self._finished:
            raise ContractError("native paired bridge is single-use")
        self._finished = True
        data = _json_bytes(request)
        if len(data) > MAX_BYTES:
            raise ContractError("paired generation request exceeds its complete bound")
        output, _ = self._process.communicate(data)
        if self._process.returncode or len(output) > MAX_BYTES:
            raise ContractError("native paired execution failed or exceeded its bounded output")
        result = strict_json(output.decode())
        if result.get("population") != self.population or result.get("request") != request:
            raise ContractError("native paired output substituted the captured population or generation request")
        return result


@contextmanager
def capture_population(gw: Path, database: Path, registration: str, split: str):
    """Start the existing local native controller and expose only its redacted complete split."""
    if split not in {"validation", "test"}:
        raise ContractError("comparison requires the complete validation or test partition")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(gw.resolve(strict=True)), "eval", "coding-pair", "--db", str(database),
                                    "--registration", registration, "--split", split, "--stdio"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   start_new_session=True)
        try:
            raw = process.stdout.readline(MAX_BYTES + 1)
            if not raw.endswith(b"\n") or len(raw) > MAX_BYTES:
                raise ContractError("native controller did not supply one bounded captured population")
            first = strict_json(raw.decode())
            if type(first) is not dict or set(first) != {"protocol_version", "population"} or first["protocol_version"] != 1:
                raise ContractError("unsupported native population bridge protocol")
            population = first["population"]
            if population.get("registration_id") != registration or population.get("split") != split:
                raise ContractError("native population differs from the requested registration and split")
            yield NativePairBridge(process, errors, population)
        finally:
            _settle(process)
            if process.stdout is not None:
                process.stdout.close()


def run_saved(gw, database, data):
    """Freshly re-execute saved modules through native private oracles without any model loading."""
    if len(data) > MAX_BYTES:
        raise ContractError("saved paired artifact exceeds bound")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(gw.resolve(strict=True)), "eval", "coding-pair-replay", "--db", str(database), "--stdin"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
        try:
            output, _ = process.communicate(data)
            if process.returncode or len(output) > MAX_BYTES:
                raise ContractError("fresh native paired replay differs from the saved declaration or failed")
            return strict_json(output.decode())
        finally:
            _settle(process)
            if process.stdout is not None:
                process.stdout.close()
