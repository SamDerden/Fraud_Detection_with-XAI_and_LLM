"""One-command Codespaces demo. Prefer verified ExtraTrees v4 artifact."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTRATREES = ROOT / "artifacts" / "extratrees-v4"
ARTIFACT = ROOT / "artifacts" / "catboost-v3"
BLEND = ROOT / "artifacts" / "blend-v3"
EXPECTED_VERSION = "catboost-educational-v3"


def prepare():
    # Prefer the final selected model, not historical CatBoost/Blend.
    archive_v4 = ROOT / "extratrees-v4-artifact.zip"
    if archive_v4.is_file():
        EXTRATREES.mkdir(parents=True, exist_ok=True)
        import shutil
        with zipfile.ZipFile(archive_v4) as z:
            for name in ("model.joblib", "metadata.json"):
                with z.open("extratrees-v4/" + name) as source, (EXTRATREES / name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        from antifraud.extratrees_v4 import load_extratrees
        load_extratrees.cache_clear()
        _, meta = load_extratrees(str(EXTRATREES))
        os.environ["ANTIFRAUD_MODEL_DIR"] = str(EXTRATREES)
        os.environ["ANTIFRAUD_MODEL_KIND"] = "extratrees_v4"
        os.environ["ANTIFRAUD_REQUIRE_ARTIFACT"] = "1"
        print("Загружена ExtraTrees v4:", meta["model_version"], flush=True)
        configure_llm()
        return
    BLEND.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "blend-v3-artifact.zip"
    names = ("catboost.cbm", "lightgbm.joblib", "metadata.json")
    if archive.is_file() and not all((BLEND / name).is_file() for name in names):
        import shutil
        with zipfile.ZipFile(archive) as z:
            for name in names:
                with z.open("blend-v3/" + name) as source, (BLEND / name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
    if all((BLEND / name).is_file() for name in names):
        from antifraud.blend import load_blend
        _, _, meta = load_blend(str(BLEND))
        os.environ["ANTIFRAUD_MODEL_DIR"] = str(BLEND)
        os.environ["ANTIFRAUD_MODEL_KIND"] = "blend"
        os.environ["ANTIFRAUD_REQUIRE_ARTIFACT"] = "1"
        print("Загружен ансамбль:", meta["model_version"], flush=True)
        configure_llm()
        return
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
    configure_llm()


def configure_llm():
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
