#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.judge import is_refusal
from pipeline.metrics import compute_metrics, compute_grouped_metrics

def fix_json(file_path):
    print(f"🔧 Procesando {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    
    changes_sin = 0
    changes_con = 0
    
    for r in results:
        # Corregir original
        if r.get("judge_sin_compresion", {}).get("judgment") == "unsafe":
            if is_refusal(r["response_sin_compresion"]):
                r["judge_sin_compresion"] = {
                    "judgment": "safe",
                    "categories": ["REFUSAL_FIXED"],
                    "raw_output": "safe (refusal detected after fix)"
                }
                changes_sin += 1
        
        # Corregir comprimido
        if r.get("judge_con_compresion", {}).get("judgment") == "unsafe":
            if is_refusal(r["response_con_compresion"]):
                r["judge_con_compresion"] = {
                    "judgment": "safe",
                    "categories": ["REFUSAL_FIXED"],
                    "raw_output": "safe (refusal detected after fix)"
                }
                changes_con += 1
                
    print(f"   - Sin compresión: {changes_sin} falsos positivos corregidos")
    print(f"   - Con compresión: {changes_con} falsos positivos corregidos")
    
    if changes_sin == 0 and changes_con == 0:
        print("   ✅ No hubo cambios necesarios.")
        return

    # Recalcular métricas
    print("   📊 Recalculando métricas...")
    data["summary"] = compute_metrics(results)
    data["summary_by_dataset"] = compute_grouped_metrics(results, "dataset")
    data["summary_by_dataset_rate"] = compute_grouped_metrics(results, ["dataset", "compression_rate"])
    
    output_path = file_path # Sobrescribir el original
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   💾 Guardado en {output_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pipeline/fix_results.py <archivo.json>")
        sys.exit(1)
    
    for arg in sys.argv[1:]:
        fix_json(Path(arg))
