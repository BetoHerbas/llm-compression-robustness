import os
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("resultados")
OUTPUT_FILE = Path("dashboard/public/data.json")

def aggregate():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} not found.")
        return

    grouped_points = {}
    models = set()
    templates = set()
    compressors = set()
    all_data = []

    # Create dashboard dir if not exists
    Path("dashboard/public").mkdir(parents=True, exist_ok=True)

    files = list(RESULTS_DIR.glob("*.json"))
    print(f"Found {len(files)} result files.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            exp_meta = data.get("experiment", {})
            model = exp_meta.get("llm_model", "Unknown")
            template = exp_meta.get("template", "raw")
            compressor = exp_meta.get("compressor_type", "llmlingua2") # Fallback for older files

            # Skip perplexity — paper compares only extractive vs abstractive
            if compressor == "perplexity":
                print(f"  Skipping {file_path.name} (compressor=perplexity)")
                continue
            
            models.add(model)
            templates.add(template)
            compressors.add(compressor)

            rates_data = data.get("summary_by_dataset_rate", {})
            
            if not rates_data and "results" in data:
                # Fallback grouping for old versions
                raw_results = data["results"]
                temp_groups = {}
                for r in raw_results:
                    g_key = f"{r['dataset']} | {r['compression_rate']}"
                    temp_groups.setdefault(g_key, []).append(r)
                
                for key, items in temp_groups.items():
                    total = len(items)
                    unsafe_sin = sum(1 for r in items if r.get("judge_sin_compresion", {}).get("judgment") == "unsafe")
                    unsafe_con = sum(1 for r in items if r.get("judge_con_compresion", {}).get("judgment") == "unsafe")
                    
                    rates_data[key] = {
                        "total_samples": total,
                        "asr_sin_compresion": unsafe_sin / total,
                        "asr_con_compresion": unsafe_con / total,
                        "delta_asr": (unsafe_con - unsafe_sin) / total,
                        "latencia_promedio_sin_compresion": sum(r.get("latency_sin_compresion", 0) for r in items) / total,
                        "latencia_promedio_con_compresion": sum(r.get("latency_con_compresion", 0) for r in items) / total,
                        "latencia_promedio_compresion_llmlingua": sum(r.get("latency_compresion", 0) for r in items) / total,
                        "tokens_promedio_original": sum(r.get("tokens_original", 0) for r in items) / total,
                        "tokens_promedio_comprimido": sum(r.get("tokens_comprimidos", 0) for r in items) / total,
                    }

            for key, m in rates_data.items():
                parts = key.split("|")
                rate = float(parts[1].strip())
                
                group_key = (model, template, compressor, rate)
                if group_key not in grouped_points:
                    grouped_points[group_key] = {
                        "total_samples": 0,
                        "sum_asr_sin": 0, "sum_asr_con": 0, "sum_delta": 0,
                        "sum_lat_sin": 0, "sum_lat_con": 0, "sum_lat_comp": 0,
                        "sum_tokens_orig": 0, "sum_tokens_comp": 0
                    }
                
                stats = grouped_points[group_key]
                samples = m.get("total_samples", 0)
                stats["total_samples"] += samples
                stats["sum_asr_sin"] += m.get("asr_sin_compresion", 0) * samples
                stats["sum_asr_con"] += m.get("asr_con_compresion", 0) * samples
                stats["sum_delta"] += m.get("delta_asr", 0) * samples
                stats["sum_lat_sin"] += m.get("latencia_promedio_sin_compresion", 0) * samples
                stats["sum_lat_con"] += m.get("latencia_promedio_con_compresion", 0) * samples
                stats["sum_lat_comp"] += m.get("latencia_promedio_compresion_llmlingua", 0) * samples
                stats["sum_tokens_orig"] += m.get("tokens_promedio_original", 0) * samples
                stats["sum_tokens_comp"] += m.get("tokens_promedio_comprimido", 0) * samples

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    # Convert to flat entries
    for (model, template, compressor, rate), stats in grouped_points.items():
        n = max(stats["total_samples"], 1)
        entry = {
            "model": model,
            "template": template,
            "compressor": compressor,
            "rate": rate,
            "asr_sin": stats["sum_asr_sin"] / n,
            "asr_con": stats["sum_asr_con"] / n,
            "delta_asr": stats["sum_delta"] / n,
            "latency_sin": stats["sum_lat_sin"] / n,
            "latency_con": stats["sum_lat_con"] / n,
            "latency_comp": stats["sum_lat_comp"] / n,
            "tokens_sin": stats["sum_tokens_orig"] / n,
            "tokens_con": stats["sum_tokens_comp"] / n,
            "ratio": (stats["sum_tokens_comp"] / n) / max(stats["sum_tokens_orig"] / n, 1),
            "total_samples": stats["total_samples"]
        }
        all_data.append(entry)

    # Calculate Unified Baseline (Pooled Average) for each (Model, Template)
    unified_baselines = {}
    baseline_counts = {}
    for entry in all_data:
        key = (entry["model"], entry["template"])
        if key not in unified_baselines:
            unified_baselines[key] = 0.0
            baseline_counts[key] = 0
        
        # Weight by total_samples to get a true pooled average
        unified_baselines[key] += entry["asr_sin"] * entry["total_samples"]
        baseline_counts[key] += entry["total_samples"]
        
    for key in unified_baselines:
        if baseline_counts[key] > 0:
            unified_baselines[key] /= baseline_counts[key]
            
    # Apply Unified Baseline to all entries
    for entry in all_data:
        key = (entry["model"], entry["template"])
        if key in unified_baselines:
            entry["asr_sin"] = unified_baselines[key]

    # Sort data for clean line representation (Model -> Template -> Compressor -> Rate DESC)
    all_data.sort(key=lambda x: (x["model"], x["template"], x["compressor"], -x["rate"]))

    result = {
        "generated_at": datetime.now().isoformat(),
        "models": sorted(list(models)),
        "templates": sorted(list(templates)),
        "compressors": sorted(list(compressors)),
        "entries": all_data
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"Successfully aggregated {len(all_data)} rate points into {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate()
