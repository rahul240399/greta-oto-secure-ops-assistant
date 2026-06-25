import os

from secureops.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.chunking.size == 1000
    assert cfg.retrieval.hybrid is True
    assert cfg.generation.model.startswith("gemini")


def test_load_config_reads_yaml(tmp_path):
    yaml_text = """
chunking:
  size: 512
  overlap: 64
retrieval:
  hybrid: false
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.chunking.size == 512
    assert cfg.chunking.overlap == 64
    assert cfg.retrieval.hybrid is False
    # untouched values keep their defaults
    assert cfg.retrieval.top_k == 5


def test_api_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    cfg = load_config(tmp_path / "missing.yaml")  # file absent -> defaults + env
    assert cfg.google_api_key == "test-key-123"
