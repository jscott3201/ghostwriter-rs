"""The actual producer's no-overwrite commit point and retained-publication error semantics."""
import os
import stat

from ..artifact import ContractError
from ..lora.safe_tensors import regular

from ..training.publication import PublishedCheckpointError


class Publication:
    """Track irreversible pathname publication separately from durability and cleanup."""
    def __init__(self, completion_id, prepared_id, output):
        self.completion_id = completion_id
        self.prepared_id = prepared_id
        self.output = output
        self.published = False
        self.directory_synced = False
        self.directory = None
        self.phase = "publication"

    def commit(self, staged):
        """Link exactly once, then sync the containing directory without replacing an existing path."""
        with regular(staged) as source:
            identity = os.fstat(source.fileno())
            try:
                os.link(staged, self.output)
                self.published = True
            except FileExistsError:
                # EEXIST proves this attempt did not link, even for an existing same-inode path.
                raise
            except BaseException as error:
                # A signal can arrive after link(2) succeeds but before Python records it.
                try:
                    target = os.stat(self.output, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                except BaseException as settlement_error:
                    raise ContractError(
                        f"CUDA publication outcome unknown for completion {self.completion_id} "
                        f"at {self.output}; target identity could not be inspected: "
                        f"{type(settlement_error).__name__}; initial failure: {type(error).__name__}"
                    ) from error
                else:
                    self.published = (stat.S_ISREG(target.st_mode)
                                      and (target.st_dev, target.st_ino) == (identity.st_dev, identity.st_ino))
                raise
        self.phase = "directory_open"
        self.directory = os.open(self.output.parent, os.O_RDONLY)
        self.phase = "directory_sync"
        os.fsync(self.directory)
        self.directory_synced = True

    def close(self):
        """Attempt descriptor cleanup separately; the published pathname is never unlinked."""
        if self.directory is not None:
            descriptor, self.directory = self.directory, None
            os.close(descriptor)

    def raise_failure(self, errors):
        """Preserve the committed artifact identity and truthful durability after any later error."""
        if self.published:
            raise PublishedCheckpointError(self.completion_id, self.prepared_id, self.output,
                                           self.directory_synced, errors) from errors[0][1]
        raise errors[0][1]
