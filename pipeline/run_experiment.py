#!/usr/bin/env python3
"""
Pipeline de Tesis: Impacto de Compresión de Prompt en Ataques de Jailbreak
══════════════════════════════════════════════════════════════════════════

Flujo:
  Prompt original
        │
        ▼
   [Opcional] Prompt Guard (detección)
        │
        ▼
   LLMLingua-2  ──>  Prompt comprimido
        │
        ▼
   Llama 4 Scout (LLM principal)
        │
        ▼
   Llama Guard 3 (judge)
        │
        ▼
   Métricas: ASR / detección / latencia / tokens

Uso:
  export TOGETHER_API_KEY="tu-api-key"
  python pipeline/run_experiment.py --num-samples 20
  python pipeline/run_experiment.py --num-samples 100 --use-prompt-guard
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
from llmlingua import PromptCompressor

# Asegurar que el directorio raíz (tests/) esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.llm_client import LLMClient
from pipeline.judge import JudgeLlamaGuard
from pipeline.metrics import compute_metrics


# ── Helpers ──────────────────────────────────────────────────────────────

def init_compressor():
    """Inicializa LLMLingua-2 en CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("   Cargando LLMLingua-2...")
    compressor = PromptCompressor(
        model_name=config.COMPRESSOR_MODEL,
        use_llmlingua2=True,
        device_map="cpu",
    )
    print("   ✔ LLMLingua-2 listo")
    return compressor


def compress_prompt(compressor, text: str) -> dict:
    """Comprime un prompt y devuelve resultado + timing."""
    start = time.time()
    result = compressor.compress_prompt_llmlingua2(
        text,
        rate=config.COMPRESSION_RATE,
        force_tokens=config.FORCE_TOKENS,
    )
    latency = time.time() - start
    return {
        "compressed_prompt": result["compressed_prompt"],
        "tokens_original": result["origin_tokens"],
        "tokens_compressed": result["compressed_tokens"],
        "ratio": result["rate"],
        "latency": latency,
    }


# ── Experimento principal ────────────────────────────────────────────────

def run_experiment(args):
    active_profile = config.PROFILE
    print("🚀 Iniciando Pipeline de Tesis")
    print(f"   Perfil:          {active_profile}")
    print(f"   Backend:         {config.BACKEND}")
    print(f"   Modelo LLM:      {config.LLM_MODEL}")
    print(f"   Modelo Judge:    {config.JUDGE_MODEL}")
    print(f"   Compresión:      {config.COMPRESSION_RATE * 100:.0f}%")
    print(f"   Muestras:        {args.num_samples}")
    print(f"   Prompt Guard:    {'Sí' if args.use_prompt_guard else 'No'}")
    print()

    # 1 ── Componentes ────────────────────────────────────────────────────
    print("📦 Inicializando componentes...")
    compressor = init_compressor()
    llm = LLMClient()
    judge = JudgeLlamaGuard()

    prompt_guard = None
    if args.use_prompt_guard:
        from pipeline.prompt_guard import PromptGuard
        print("   Cargando Prompt Guard...")
        prompt_guard = PromptGuard()
        print("   ✔ Prompt Guard listo")

    print()

    # 2 ── Dataset ────────────────────────────────────────────────────────
    print("📦 Cargando JailbreakBench...")
    dataset = load_dataset(
        config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT
    )
    num_samples = min(args.num_samples, len(dataset))
    print(f"   ✔ {len(dataset)} ataques disponibles, usando {num_samples}")
    print()

    # 3 ── Bucle principal ────────────────────────────────────────────────
    results = []
    errors = []

    print("🔬 Ejecutando experimento...")
    for i in tqdm(range(num_samples), desc="Procesando"):
        item = dataset[i]
        prompt_original = item["Goal"]

        entry = {
            "id": item["Index"],
            "categoria": item["Category"],
            "comportamiento": item["Behavior"],
            "prompt_original": prompt_original,
        }

        try:
            # 3a ── Prompt Guard sobre original ───────────────────────────
            if prompt_guard:
                entry["prompt_guard_original"] = prompt_guard.detect(prompt_original)
            else:
                entry["prompt_guard_original"] = None

            # 3b ── Compresión ────────────────────────────────────────────
            comp = compress_prompt(compressor, prompt_original)
            entry["prompt_comprimido"] = comp["compressed_prompt"]
            entry["tokens_original"] = comp["tokens_original"]
            entry["tokens_comprimidos"] = comp["tokens_compressed"]
            entry["ratio_compresion"] = comp["ratio"]
            entry["latency_compresion"] = round(comp["latency"], 4)

            # 3c ── Prompt Guard sobre comprimido ─────────────────────────
            if prompt_guard:
                entry["prompt_guard_compressed"] = prompt_guard.detect(
                    comp["compressed_prompt"]
                )
            else:
                entry["prompt_guard_compressed"] = None

            # 3d ── LLM sin compresión ────────────────────────────────────
            resp_sin = llm.generate(prompt_original)
            entry["response_sin_compresion"] = resp_sin["response"]
            entry["latency_sin_compresion"] = round(resp_sin["latency"], 4)
            entry["tokens_llm_sin_compresion"] = resp_sin["tokens_completion"]

            # 3e ── LLM con compresión ────────────────────────────────────
            resp_con = llm.generate(comp["compressed_prompt"])
            entry["response_con_compresion"] = resp_con["response"]
            entry["latency_con_compresion"] = round(resp_con["latency"], 4)
            entry["tokens_llm_con_compresion"] = resp_con["tokens_completion"]

            # 3f ── Judge sin compresión ──────────────────────────────────
            entry["judge_sin_compresion"] = judge.evaluate(
                prompt_original, resp_sin["response"]
            )

            # 3g ── Judge con compresión ──────────────────────────────────
            entry["judge_con_compresion"] = judge.evaluate(
                comp["compressed_prompt"], resp_con["response"]
            )

            results.append(entry)

        except Exception as e:
            errors.append({"id": item["Index"], "error": str(e)})
            tqdm.write(f"   ⚠ Error en muestra {item['Index']}: {e}")

        # Rate-limit para no saturar la API
        time.sleep(args.delay)

    # 4 ── Métricas ───────────────────────────────────────────────────────
    print("\n📊 Calculando métricas...")
    summary = compute_metrics(results) if results else {}

    # 5 ── Guardar resultados ─────────────────────────────────────────────
    output = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "compressor_model": config.COMPRESSOR_MODEL,
            "compression_rate": config.COMPRESSION_RATE,
            "system_prompt": config.SYSTEM_PROMPT,
            "num_samples_requested": args.num_samples,
            "num_processed": len(results),
            "num_errors": len(errors),
            "use_prompt_guard": args.use_prompt_guard,
        },
        "summary": summary,
        "results": results,
        "errors": errors,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.OUTPUT_DIR / f"experiment_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 6 ── Resumen en consola ─────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  📋  RESUMEN DEL EXPERIMENTO")
    print("=" * 62)
    print(f"  Muestras procesadas:       {summary.get('total_samples', 0)}")
    print(f"  Errores:                   {len(errors)}")
    print()
    print(f"  ASR sin compresión:        {summary.get('asr_sin_compresion', 0) * 100:.1f}%")
    print(f"  ASR con compresión:        {summary.get('asr_con_compresion', 0) * 100:.1f}%")
    print(f"  Δ ASR:                     {summary.get('delta_asr', 0) * 100:+.1f}%")
    print()
    print(f"  Latencia prom. (sin):      {summary.get('latencia_promedio_sin_compresion', 0):.2f}s")
    print(f"  Latencia prom. (con):      {summary.get('latencia_promedio_con_compresion', 0):.2f}s")
    print(f"  Latencia compresión:       {summary.get('latencia_promedio_compresion_llmlingua', 0):.2f}s")
    print()
    print(f"  Tokens prom. original:     {summary.get('tokens_promedio_original', 0)}")
    print(f"  Tokens prom. comprimido:   {summary.get('tokens_promedio_comprimido', 0)}")
    print(f"  Ratio compresión prom.:    {summary.get('ratio_compresion_promedio', 0) * 100:.1f}%")

    if args.use_prompt_guard and "deteccion_prompt_guard_original" in summary:
        print()
        print(f"  Detección PG (original):   {summary['deteccion_prompt_guard_original'] * 100:.1f}%")
        print(f"  Detección PG (comprimido): {summary['deteccion_prompt_guard_comprimido'] * 100:.1f}%")
        print(f"  Evasión por compresión:    {summary['evasion_por_compresion'] * 100:.1f}%")

    print()
    print(f"  Resultados en: {output_file}")
    print("=" * 62)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Tesis – Compresión de Prompt vs Jailbreak"
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "lab"],
        default=config.PROFILE.lower(),
        help=(
            "Perfil de ejecución: pilot (laptop) o lab (final). "
            f"Default: {config.PROFILE.lower()}"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default=None,
        help="Backend para inferencia: local (Transformers) o api (OpenAI-compatible)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help=f"Número de muestras (default: {config.NUM_SAMPLES})",
    )
    parser.add_argument(
        "--use-prompt-guard", action="store_true", default=None,
        help="Activar detección con Prompt Guard (local, ~86M params)",
    )
    parser.add_argument(
        "--no-prompt-guard", action="store_true",
        help="Desactivar Prompt Guard aunque el perfil lo tenga activo",
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="Segundos entre llamadas al API (rate-limit, default: 1.0)",
    )
    args = parser.parse_args()

    # Aplicar perfil antes de inicializar clientes/modelos.
    selected = config.apply_profile(args.profile)

    if args.backend is not None:
        config.BACKEND = args.backend
    else:
        config.BACKEND = selected["BACKEND"].lower()

    if args.num_samples is None:
        args.num_samples = selected["NUM_SAMPLES"]

    if args.delay is None:
        args.delay = selected["DELAY"]

    # Precedencia: --no-prompt-guard > --use-prompt-guard > perfil
    if args.no_prompt_guard:
        args.use_prompt_guard = False
    elif args.use_prompt_guard is None:
        args.use_prompt_guard = selected["USE_PROMPT_GUARD"]

    if config.BACKEND == "api" and not config.API_KEY:
        print("❌ Error: TOGETHER_API_KEY no está configurada.")
        print("   Ejecuta:  export TOGETHER_API_KEY='tu-api-key'")
        print("   O regístrate gratis en https://api.together.xyz")
        sys.exit(1)

    run_experiment(args)


if __name__ == "__main__":
    main()
