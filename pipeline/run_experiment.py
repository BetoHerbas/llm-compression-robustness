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
from pipeline.metrics import compute_metrics, compute_grouped_metrics


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


def compress_prompt(compressor, text: str, rate: float) -> dict:
    """Comprime un prompt y devuelve resultado + timing."""
    start = time.time()
    result = compressor.compress_prompt_llmlingua2(
        text,
        rate=rate,
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


def parse_csv_list(raw: str | None, cast=str) -> list:
    if raw is None:
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def load_dataset_by_alias(alias: str):
    spec = config.DATASET_REGISTRY.get(alias)
    if spec is None:
        raise ValueError(f"Dataset alias no soportado: {alias}")

    if spec["hf_config"] is None:
        ds = load_dataset(spec["hf_name"], split=spec["split"])
    else:
        ds = load_dataset(spec["hf_name"], spec["hf_config"], split=spec["split"])
    return ds, spec


# ── Experimento principal ────────────────────────────────────────────────

def run_experiment(args):
    active_profile = config.PROFILE
    print("🚀 Iniciando Pipeline de Tesis")
    print(f"   Perfil:          {active_profile}")
    print(f"   Backend:         {config.BACKEND}")
    print(f"   Modelo LLM:      {config.LLM_MODEL}")
    print(f"   Modelo Judge:    {config.JUDGE_MODEL}")
    print(f"   Rates:           {', '.join(str(r) for r in args.rates)}")
    print(f"   Datasets:        {', '.join(args.datasets)}")
    print(f"   Muestras/dataset:{args.num_samples}")
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
    datasets_loaded = {}
    dataset_load_errors = []
    print("📦 Cargando datasets...")
    for alias in args.datasets:
        try:
            ds, spec = load_dataset_by_alias(alias)
            num_samples = min(args.num_samples, len(ds))
            datasets_loaded[alias] = {
                "dataset": ds,
                "spec": spec,
                "num_samples": num_samples,
            }
            print(f"   ✔ {alias}: {len(ds)} filas, usando {num_samples}")
        except Exception as e:
            dataset_load_errors.append({"dataset": alias, "error": str(e)})
            print(f"   ⚠ {alias}: no se pudo cargar ({e})")

    if not datasets_loaded:
        raise RuntimeError("No se pudo cargar ningún dataset.")
    print()

    # 3 ── Bucle principal ────────────────────────────────────────────────
    results = []
    errors = []

    print("🔬 Ejecutando experimento...")
    for dataset_alias, payload in datasets_loaded.items():
        ds = payload["dataset"]
        spec = payload["spec"]
        n = payload["num_samples"]

        for rate in args.rates:
            desc = f"{dataset_alias} @ rate={rate}"
            for i in tqdm(range(n), desc=desc):
                item = ds[i]
                prompt_field = spec["prompt_field"]
                prompt_original = item[prompt_field]

                idx_field = spec["id_field"]
                cat_field = spec["category_field"]
                beh_field = spec["behavior_field"]
                source_id = item[idx_field] if idx_field and idx_field in item else i
                rate_tag = str(rate).replace(".", "p")
                unique_id = f"{dataset_alias}__r{rate_tag}__i{i}__src{source_id}"

                entry = {
                    "dataset": dataset_alias,
                    "compression_rate": rate,
                    "id": unique_id,
                    "source_id": source_id,
                    "sample_index": i,
                    "categoria": item[cat_field] if cat_field and cat_field in item else "N/A",
                    "comportamiento": item[beh_field] if beh_field and beh_field in item else "N/A",
                    "prompt_original": prompt_original,
                }

                try:
                    if prompt_guard:
                        entry["prompt_guard_original"] = prompt_guard.detect(prompt_original)
                    else:
                        entry["prompt_guard_original"] = None

                    comp = compress_prompt(compressor, prompt_original, rate)
                    entry["prompt_comprimido"] = comp["compressed_prompt"]
                    entry["tokens_original"] = comp["tokens_original"]
                    entry["tokens_comprimidos"] = comp["tokens_compressed"]
                    entry["ratio_compresion"] = comp["ratio"]
                    entry["latency_compresion"] = round(comp["latency"], 4)

                    if prompt_guard:
                        entry["prompt_guard_compressed"] = prompt_guard.detect(
                            comp["compressed_prompt"]
                        )
                    else:
                        entry["prompt_guard_compressed"] = None

                    resp_sin = llm.generate(prompt_original)
                    entry["response_sin_compresion"] = resp_sin["response"]
                    entry["latency_sin_compresion"] = round(resp_sin["latency"], 4)
                    entry["tokens_llm_sin_compresion"] = resp_sin["tokens_completion"]

                    resp_con = llm.generate(comp["compressed_prompt"])
                    entry["response_con_compresion"] = resp_con["response"]
                    entry["latency_con_compresion"] = round(resp_con["latency"], 4)
                    entry["tokens_llm_con_compresion"] = resp_con["tokens_completion"]

                    entry["judge_sin_compresion"] = judge.evaluate(
                        prompt_original, resp_sin["response"]
                    )

                    entry["judge_con_compresion"] = judge.evaluate(
                        comp["compressed_prompt"], resp_con["response"]
                    )

                    results.append(entry)

                except Exception as e:
                    errors.append(
                        {
                            "dataset": dataset_alias,
                            "compression_rate": rate,
                            "id": entry["id"],
                            "error": str(e),
                        }
                    )
                    tqdm.write(
                        f"   ⚠ Error en {dataset_alias} muestra {entry['id']} (rate={rate}): {e}"
                    )

                time.sleep(args.delay)

    # 4 ── Métricas ───────────────────────────────────────────────────────
    print("\n📊 Calculando métricas...")
    summary = compute_metrics(results) if results else {}
    by_dataset = compute_grouped_metrics(results, "dataset")
    by_dataset_rate = compute_grouped_metrics(results, ["dataset", "compression_rate"])

    # 5 ── Guardar resultados ─────────────────────────────────────────────
    output = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "compressor_model": config.COMPRESSOR_MODEL,
            "compression_rate": config.COMPRESSION_RATE,
            "compression_rates": args.rates,
            "datasets": args.datasets,
            "system_prompt": config.SYSTEM_PROMPT,
            "num_samples_requested": args.num_samples,
            "num_processed": len(results),
            "num_errors": len(errors),
            "use_prompt_guard": args.use_prompt_guard,
        },
        "summary": summary,
        "summary_by_dataset": by_dataset,
        "summary_by_dataset_rate": by_dataset_rate,
        "results": results,
        "errors": errors,
        "dataset_load_errors": dataset_load_errors,
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
    print(f"  McNemar p-value:           {summary.get('mcnemar_pvalue', 1.0)}")

    if by_dataset:
        print()
        print("  ΔASR por dataset:")
        for dataset_name, m in by_dataset.items():
            print(
                f"    - {dataset_name}: {m.get('delta_asr', 0) * 100:+.1f}% "
                f"(p={m.get('mcnemar_pvalue', 1.0)})"
            )

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
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Lista CSV de datasets alias (e.g. jbb,advbench)",
    )
    parser.add_argument(
        "--rates", type=str, default=None,
        help="Lista CSV de rates de compresión (e.g. 0.9,0.7,0.5,0.3)",
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

    if args.datasets is None:
        args.datasets = config.ACTIVE_DATASETS
    else:
        args.datasets = parse_csv_list(args.datasets, cast=str)

    if args.rates is None:
        args.rates = config.COMPRESSION_RATES
    else:
        args.rates = parse_csv_list(args.rates, cast=float)

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
