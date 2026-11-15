"""Every environment setting the package reads is declared once, in fpl_model/config.py."""
import re
from pathlib import Path

import pytest

from fpl_model import config

PACKAGE = Path(__file__).resolve().parents[1] / "fpl_model"
# Reading the environment directly: .get, subscripts, membership tests, getenv. Writes (setdefault) and passing the
# whole environment to a subprocess ({**os.environ, ...}) are not reads of a setting.
DIRECT_READ = re.compile(r"os\.environ\.get\(|os\.environ\[|\bin\s+os\.environ\b|os\.getenv\("
                         r"|__import__\(\"os\"\)\.environ")
CONFIG_CALL = re.compile(r"config\.(?:get|is_set)\(\s*[\"']([A-Z0-9_]+)[\"']")
# The vendored solver keeps its upstream code.
EXEMPT = {"config.py", "optimizer/open_fpl_solver"}


def _package_files():
    for p in sorted(PACKAGE.rglob("*.py")):
        rel = p.relative_to(PACKAGE).as_posix()
        if not any(rel == e or rel.startswith(e + "/") for e in EXEMPT):
            yield rel, p.read_text(encoding="utf-8")


def test_no_direct_environment_reads():
    offenders = [f"{rel}:{i}" for rel, text in _package_files()
                 for i, line in enumerate(text.splitlines(), 1) if DIRECT_READ.search(line)]
    assert not offenders, "read settings through fpl_model.config, not os.environ: " + ", ".join(offenders)


def test_every_setting_used_is_declared():
    used = {name for _, text in _package_files() for name in CONFIG_CALL.findall(text)}
    assert used, "no config.get calls found; the scan is broken"
    assert used <= set(config.SETTINGS), f"undeclared settings: {sorted(used - set(config.SETTINGS))}"


def test_every_setting_is_documented():
    for s in config.SETTINGS.values():
        assert s.meaning and s.changelog.startswith("§"), s.name
        assert s.kind in ("bool", "int", "float", "str", "path", "choice"), s.name
        if s.kind == "choice":
            assert s.default in s.choices, s.name


def test_defaults_and_parsing(monkeypatch):
    for name in config.SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert config.get("GOAL_MODEL_ALPHA_NPG") == 0.035
    assert config.get("GOAL_MODEL_ALPHA_NPG_CB") == 0.018
    assert config.get("DEFCON_POINTS_CALIBRATION") is True
    assert config.get("GOAL_MODEL_PREMATCH_LOG") == ""
    # an experiment that moves the general floor moves the npxG and centre-back floors too (pre-§233/§235 behaviour)
    monkeypatch.setenv("GOAL_MODEL_ALPHA", "0.05")
    assert config.get("GOAL_MODEL_ALPHA_NPG") == config.get("GOAL_MODEL_ALPHA_NPG_CB") == 0.05
    monkeypatch.setenv("GOAL_MODEL_PRIOR_K", "5")
    assert config.get("GOAL_MODEL_PRIOR_K_YOUTH") == 5.0
    monkeypatch.setenv("BONUS_SAVE_FIXTURE", "0")
    assert config.get("BONUS_SAVE_FIXTURE") is False
    monkeypatch.setenv("BONUS_SAVE_FIXTURE", "yes")
    with pytest.raises(ValueError):
        config.get("BONUS_SAVE_FIXTURE")
    monkeypatch.setenv("CARD_ROLE_MIX", "nonsense")
    with pytest.raises(ValueError):
        config.get("CARD_ROLE_MIX")
    with pytest.raises(KeyError):
        config.get("NOT_A_SETTING")
