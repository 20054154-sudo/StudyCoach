"""Configuration files keep secrets separate from application defaults."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_env_example_contains_only_api_key_names():
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    names = {line.split("=", 1)[0] for line in lines if line.strip() and not line.startswith("#")}
    assert names == {"GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY"}
    assert all(not line.split("=", 1)[1] for line in lines if "=" in line)


def test_nonsecret_defaults_are_in_toml():
    with (PROJECT_ROOT / "config" / "app.toml").open("rb") as stream:
        settings = tomllib.load(stream)
    assert settings["RAG_CHUNK_WORDS"] == 90
    assert settings["QUIZ_PROVIDER"] == "mistral"
    assert not any(name.endswith("API_KEY") for name in settings)


def test_process_environment_overrides_app_config():
    env = os.environ.copy()
    env["RAG_CHUNK_WORDS"] = "123"
    result = subprocess.run(
        [sys.executable, "-c", "import backend, os; print(os.environ['RAG_CHUNK_WORDS'])"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "123"
