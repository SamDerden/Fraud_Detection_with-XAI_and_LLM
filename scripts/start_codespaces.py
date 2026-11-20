"""One-command Codespaces demo. Requires trusted v3 model archive, never uses toy RF."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "catboost-v3"
EXPECTED_VERSION = "catboost-educational-v3"


def prepare():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    if not ((ARTIFACT / "model.cbm").is_file() and (ARTIFACT / "metadata.json").is_file()):
        candidates = [ROOT / "catboost-v3-artifact.zip", ROOT / "artifacts" / "catboost-v3-artifact.zip"]
        archive = next((p for p in candidates if p.is_file()), None)
        if archive is None:
            raise SystemExit(
                "\nМОДЕЛЬ V3 НЕ НАЙДЕНА.\n"
                "Загрузите catboost-v3-artifact.zip в корень Codespaces (Explorer → Upload Files).\n"
                "Затем повторите: python scripts/start_codespaces.py\n"
                "Синтетическая RandomForest НЕ будет запущена.\n"
            )
        with zipfile.ZipFile(archive) as z:
            for name, target in [("catboost-v3/model.cbm", "model.cbm"),
                                 ("catboost-v3/metadata.json", "metadata.json")]:
                with z.open(name) as src, (ARTIFACT / target).open("wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    meta = json.loads((ARTIFACT / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("model_version") != EXPECTED_VERSION:
        raise SystemExit("Неверная версия модели: ожидается catboost-educational-v3")
    from antifraud.model_artifact import load_artifact
    load_artifact(str(ARTIFACT))
    print("Загружена модель:", meta["model_version"], "|", meta["model_type"], flush=True)
    os.environ["ANTIFRAUD_MODEL_DIR"] = str(ARTIFACT)
    os.environ["ANTIFRAUD_REQUIRE_ARTIFACT"] = "1"
    if os.getenv("GROQ_API_KEY"):
        os.environ.setdefault("ANTIFRAUD_LLM_ENABLED", "true")
        os.environ.setdefault("ANTIFRAUD_LLM_PROVIDER", "groq")
        os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
        print("GroqCloud включён; внешний запрос будет при оценке транзакции", flush=True)
    else:
        print("GroqCloud: ключ не обнаружен; SHAP-объяснение работает локально", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    prepare()
    subprocess.run([sys.executable, "-m", "uvicorn", "antifraud.app:app",
                    "--host", "0.0.0.0", "--port", "8000"], env=os.environ, check=True)
