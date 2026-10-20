"""Load local application settings before importing backend components."""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The private .env contains API keys. Existing non-secret overrides remain
# supported for local installations and deployment environments.
load_dotenv(_PROJECT_ROOT / ".env", override=False)

with (_PROJECT_ROOT / "config" / "app.toml").open("rb") as _settings_file:
    _settings = tomllib.load(_settings_file)

for _name, _value in _settings.items():
    if not isinstance(_name, str) or not _name.isupper():
        raise ValueError(f"Nome di configurazione non valido: {_name!r}")
    if not isinstance(_value, (str, int, float, bool)):
        raise ValueError(f"Valore di configurazione non valido: {_name}")
    os.environ.setdefault(_name, str(_value))
