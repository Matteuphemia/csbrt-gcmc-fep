"""Minimal nbformat-4.5 notebook writer used to build the handover notebooks.

Kept dependency-free (stdlib json only) so it runs in any environment.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _split(src: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    src = src.strip("\n")
    lines = src.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _split(source),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [],
        "source": _split(source),
    }


def write_nb(path: str | Path, cells: list[dict], *, kernel_display: str = "Python 3") -> Path:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": kernel_display,
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return path
