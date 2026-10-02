"""The real public CLI consumes bounds and host policy before opening source paths."""
from ghostwriter_trl.cuda_lora.cli import main


def test_cli_rejects_unsupported_host_without_publication(tmp_path, monkeypatch, capsys):
    import ghostwriter_trl.cuda_lora.runtime as runtime
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    result = main(["train", "--prepared", str(tmp_path / "missing"), "--release-directory", str(tmp_path / "source"),
                   "--gw", str(tmp_path / "gw"), "--output", str(tmp_path / "out")])
    assert result == 2
    output = capsys.readouterr()
    assert "one Linux" in output.err and not output.out
    assert not (tmp_path / "out").exists()


def test_cli_rejects_unbounded_recipe_before_source_acquisition(tmp_path, capsys):
    result = main(["train", "--prepared", str(tmp_path / "missing"), "--release-directory", str(tmp_path / "source"),
                   "--gw", str(tmp_path / "gw"), "--output", str(tmp_path / "out"), "--max-steps", "1000"])
    assert result == 2 and "bounds" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
