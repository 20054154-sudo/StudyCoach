## Requisiti

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- Un file `.env` nella root con le sole chiavi dei provider AI selezionati (copia `.env.example`)
- Le impostazioni non segrete sono in `config/app.toml`; non occorre copiarlo

## Installazione

Dalla root del progetto:

```powershell
uv sync
```

Per eseguire anche i test:

```powershell
uv sync --extra dev
```

## Avvio

Un solo terminale, dalla root:

```powershell
uv run uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Apri **http://127.0.0.1:8000/** per l'interfaccia completa. 

## Test

```powershell
uv run --extra dev python -m pytest -q
uv lock --check
```
