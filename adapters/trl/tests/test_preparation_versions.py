"""Historical bytes retain native identity; a different installed producer cannot replay them."""
import json
import subprocess

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared


@pytest.mark.parametrize("name", ["prepared-all.gwsft", "prepared-empty.gwsft", "prepared-long.gwsft"])
def test_historical_recipe_remains_inspectable_without_changing_its_identity(name, historical_fixture_dir, gw, tokenizer):
    path = historical_fixture_dir / name
    before = path.read_bytes()
    native = subprocess.run([str(gw), "artifact", "verify-prepared", "--stdin"], input=before, capture_output=True)
    assert native.returncode == 0, native.stderr.decode()
    assert json.loads(native.stdout)["build_id"] == before[8:40].hex()
    with pytest.raises(ContractError, match="incompatible installed preparation recipe/source"):
        read_prepared(path, gw, tokenizer)
    assert path.read_bytes() == before
