"""Run common authority, complete-replay, mutation, and real long-dataloader tests on Gemma."""
from ..test_prepared_authority import (
    test_complete_loaded_long_build_reaches_real_collator_and_trainer_without_execution,
    test_replay_preserves_historical_producer_runtime_and_identity,
    test_verified_state_is_opaque_and_all_metadata_is_defensive,
)
from ..test_prepared_integrity import (
    test_self_consistent_removed_target_cannot_bypass_source_partition,
    test_shared_independent_corpus,
)
from ..test_preparation_versions import test_historical_recipe_remains_inspectable_without_changing_its_identity
