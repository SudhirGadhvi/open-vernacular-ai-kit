from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

from .transliterate import transliteration_backend


def _dist_version(dist_name: str) -> Optional[str]:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def collect_doctor_info() -> dict[str, object]:
    """
    Collect a best-effort environment report for `gck doctor`.

    This should stay lightweight and side-effect free (no network, no downloads).
    """

    # Dist names (PyPI) may differ from import names.
    optional_dists: dict[str, str] = {
        # Core
        "regex": "regex",
        "typer": "typer",
        "rich": "rich",
        # Optional features
        "indic-nlp-library": "indic-nlp-library",
        "indic-transliteration": "indic-transliteration",
        "ai4bharat-transliteration": "ai4bharat-transliteration",
        "numpy": "numpy",
        "scikit-learn": "scikit-learn",
        "joblib": "joblib",
        "sarvamai": "sarvamai",
        "python-dotenv": "python-dotenv",
        "streamlit": "streamlit",
        "pandas": "pandas",
        "requests": "requests",
        "datasets": "datasets",
        "wordfreq": "wordfreq",
        "transformers": "transformers",
        "torch": "torch",
        "pytest": "pytest",
        "ruff": "ruff",
    }

    packages: dict[str, dict[str, object]] = {}
    for name, dist in optional_dists.items():
        v = _dist_version(dist)
        packages[name] = {"installed": v is not None, "version": v}

    lid_model_path = Path(__file__).with_name("_data") / "latin_lid.joblib"

    return {
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "features": {
            "transliteration_backend": transliteration_backend(),
            "indicnlp_importable": _module_importable("indicnlp"),
            "lid_ml_model_present": lid_model_path.exists(),
        },
        "packages": packages,
    }

