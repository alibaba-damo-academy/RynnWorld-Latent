import subprocess
import sys
from pathlib import Path

import pytest

from rynnlam.config import RynnLAMConfig

ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rynnlam import RynnLAMConfig; "
            "assert 'torch' not in sys.modules; assert 'yaml' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_parsing_and_typo_detection():
    config = RynnLAMConfig.from_dict(
        {"training": {"lr": "1e-5"}, "reset_optimizer": True}
    )
    assert config.lr == 1e-5 and config.reset_optimizer
    assert RynnLAMConfig.from_dict({"old_unused_field": 1}).latent_dim == 64
    with pytest.raises(ValueError, match="Unknown"):
        RynnLAMConfig.from_dict({"training": {"lern_rate": 1}}, strict=True)
    with pytest.raises(ValueError, match="boolean"):
        RynnLAMConfig.from_dict({"reset_optimizer": "false"})
