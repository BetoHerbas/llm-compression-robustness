"""
Configuración central del pipeline de tesis.
Ajustar las variables de entorno o los valores aquí antes de correr el experimento.
"""
import os
from pathlib import Path

# ── Backend y API opcional (si usas proveedor remoto) ───────────────────
BACKEND = os.environ.get("PIPELINE_BACKEND", "local").lower()
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.together.xyz/v1")

# ── Perfil de ejecución ──────────────────────────────────────────────────
# PILOT: prueba rápida en laptop
# LAB: experimento final en laboratorio
PROFILE = os.environ.get("PIPELINE_PROFILE", "PILOT").upper()

# ── Modelos ──────────────────────────────────────────────────────────────
PROMPT_GUARD_MODEL = "meta-llama/Prompt-Guard-86M"
COMPRESSOR_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

PROFILE_SETTINGS = {
    "PILOT": {
        # Liviano para validar end-to-end en laptop.
        "BACKEND": "local",
        "LLM_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
        "JUDGE_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
        "NUM_SAMPLES": 10,
        "DELAY": 0.3,
        "USE_PROMPT_GUARD": False,
    },
    "LAB": {
        # Configuración final local para laboratorio.
        "BACKEND": "local",
        "LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        "JUDGE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
        "NUM_SAMPLES": 100,
        "DELAY": 1.0,
        "USE_PROMPT_GUARD": True,
    },
}


def _resolve_profile(profile: str) -> str:
    profile = (profile or "PILOT").upper()
    if profile not in PROFILE_SETTINGS:
        return "PILOT"
    return profile


def get_profile_settings(profile: str | None = None) -> dict:
    """Devuelve settings del perfil con override opcional por variables de entorno."""
    selected = _resolve_profile(profile or PROFILE)
    base = PROFILE_SETTINGS[selected].copy()

    # Overrides para facilitar pruebas sin editar código.
    base["BACKEND"] = os.environ.get("PIPELINE_BACKEND", base["BACKEND"])
    base["LLM_MODEL"] = os.environ.get("LLM_MODEL", base["LLM_MODEL"])
    base["JUDGE_MODEL"] = os.environ.get("JUDGE_MODEL", base["JUDGE_MODEL"])
    base["NUM_SAMPLES"] = int(os.environ.get("NUM_SAMPLES", base["NUM_SAMPLES"]))
    base["DELAY"] = float(os.environ.get("REQUEST_DELAY", base["DELAY"]))
    base["USE_PROMPT_GUARD"] = os.environ.get(
        "USE_PROMPT_GUARD", str(base["USE_PROMPT_GUARD"])
    ).lower() in ("1", "true", "yes", "y")
    base["PROFILE"] = selected
    return base


def apply_profile(profile: str | None = None) -> dict:
    """Aplica el perfil al módulo para que el runner use una única fuente."""
    selected = get_profile_settings(profile)

    global PROFILE
    global BACKEND
    global LLM_MODEL
    global JUDGE_MODEL
    global NUM_SAMPLES
    global REQUEST_DELAY
    global DEFAULT_USE_PROMPT_GUARD

    PROFILE = selected["PROFILE"]
    BACKEND = selected["BACKEND"].lower()
    LLM_MODEL = selected["LLM_MODEL"]
    JUDGE_MODEL = selected["JUDGE_MODEL"]
    NUM_SAMPLES = selected["NUM_SAMPLES"]
    REQUEST_DELAY = selected["DELAY"]
    DEFAULT_USE_PROMPT_GUARD = selected["USE_PROMPT_GUARD"]
    return selected

# ── System prompt para el LLM principal ──────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."

# ── Compresión ───────────────────────────────────────────────────────────
COMPRESSION_RATE = 0.5
FORCE_TOKENS = ['\n', '.', '!', '?', ',']

# ── Dataset ──────────────────────────────────────────────────────────────
DATASET_NAME = "JailbreakBench/JBB-Behaviors"
DATASET_CONFIG = "behaviors"
DATASET_SPLIT = "harmful"

# Variables dinámicas aplicadas por perfil
BACKEND = "local"
LLM_MODEL = ""
JUDGE_MODEL = ""
NUM_SAMPLES = 0
REQUEST_DELAY = 1.0
DEFAULT_USE_PROMPT_GUARD = False

# Inicializa defaults al importar el módulo
apply_profile(PROFILE)

# ── Salida ───────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "resultados"
OUTPUT_DIR.mkdir(exist_ok=True)
