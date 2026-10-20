"""Validated tuning values for retrieval and generation."""

import os


def positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un intero positivo.") from exc
    if value <= 0:
        raise ValueError(f"{name} deve essere un intero positivo.")
    return value


def nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un intero non negativo.") from exc
    if value < 0:
        raise ValueError(f"{name} deve essere un intero non negativo.")
    return value
