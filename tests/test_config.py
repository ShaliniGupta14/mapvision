"""Tests for YAML config loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import Config, load_config


class TestLoadConfig:
    def test_defaults_without_path(self):
        assert load_config() == Config()

    def test_default_yaml_loads(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.data.image_size == 256
        assert cfg.model.num_classes == 5

    def test_partial_override_keeps_defaults(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("training:\n  epochs: 3\n")
        cfg = load_config(str(path))
        assert cfg.training.epochs == 3
        assert cfg.training.learning_rate == Config().training.learning_rate

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("does/not/exist.yaml")

    def test_unknown_key_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("training:\n  epochz: 3\n")
        with pytest.raises(ValueError, match="epochz"):
            load_config(str(path))

    def test_unknown_section_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("optimiser:\n  lr: 1\n")
        with pytest.raises(ValueError, match="optimiser"):
            load_config(str(path))
