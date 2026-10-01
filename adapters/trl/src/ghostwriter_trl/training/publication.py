"""Explicit post-publication failures; a retained complete file is not a successful acknowledgment."""
import json

from ..artifact import ContractError, strict_json


class PublishedCheckpointError(ContractError):
    """A complete checkpoint was linked, but synchronization or resource cleanup failed.

    This diagnostic carries the verified identity for inspection. It never supplies fresh
    observed-training authority, and the destination is never rolled back or overwritten.
    """

    def __init__(self, completion_id, prepared_build_id, output, directory_synced, errors):
        report = {"report_version": 1,
                  "status": "published_cleanup_failed" if directory_synced else "published_durability_unknown",
                  "completion_id": completion_id, "prepared_build_id": prepared_build_id,
                  "output": str(output), "publication": "complete_checkpoint_linked",
                  "durability": "confirmed" if directory_synced else "unknown",
                  "errors": [{"phase": phase, "kind": type(error).__name__} for phase, error in errors]}
        self.__report = json.dumps(report)
        message = ("checkpoint was published and synchronized, but cleanup failed" if directory_synced
                   else "checkpoint was published, but its durability could not be confirmed")
        super().__init__(message)

    @property
    def report(self):
        """Detached publication diagnostic; inspect the target against this completion identity."""
        return strict_json(self.__report)

    def with_cleanup_error(self, phase, error):
        """Preserve an existing publication outcome when a surrounding resource also fails."""
        report = self.report
        report["errors"].append({"phase": phase, "kind": type(error).__name__})
        result = PublishedCheckpointError(report["completion_id"], report["prepared_build_id"],
                                          report["output"], report["durability"] == "confirmed", [])
        result.__report = json.dumps(report)
        return result


def publication_cause(error):
    """Recover the recorded outcome if an enclosing context manager masks it during cleanup."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, PublishedCheckpointError):
            return error
        seen.add(id(error))
        error = error.__context__
    return None
