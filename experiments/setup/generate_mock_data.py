import json
import os
from datetime import datetime

def create_mock_data(model, compressor, template, asr_values):
    data = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "llm_model": model,
            "judge_model": "llama-guard3",
            "template": template,
            "compressor_model": "mock",
            "compressor_type": compressor,
            "quantize_4bit": False,
            "compression_rate": 0.5,
            "compression_rates": [0.9, 0.7, 0.5, 0.3, 0.1],
            "datasets": ["jbb"]
        },
        "summary_by_dataset_rate": {}
    }

    rates = [0.9, 0.7, 0.5, 0.3, 0.1]
    
    for i, rate in enumerate(rates):
        key = f"jbb | {rate}"
        # ASR original starts at 25%
        asr_orig = 0.25
        asr_comp = asr_values[i]
        
        # Make tokens proportional to rate
        tokens_orig = 500
        tokens_comp = int(500 * rate)
        
        data["summary_by_dataset_rate"][key] = {
            "total_samples": 100,
            "asr_sin_compresion": asr_orig,
            "asr_con_compresion": asr_comp,
            "delta_asr": asr_comp - asr_orig,
            "latencia_promedio_sin_compresion": 15.0,
            "latencia_promedio_con_compresion": 15.0 * rate + 2.0,
            "latencia_promedio_compresion_llmlingua": 1.5,
            "tokens_promedio_original": tokens_orig,
            "tokens_promedio_comprimido": tokens_comp,
            "ratio_compresion_promedio": rate
        }
        
    filename = f"resultados/experiment_mock_{compressor}_{template}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Created {filename}")

# LLMLingua-2: Parabola peaking at 50% compression, very resilient
llmlingua_asr = [0.30, 0.38, 0.40, 0.38, 0.30]
create_mock_data("llama4:scout", "llmlingua2", "dev_mode", llmlingua_asr)

# Perplexity: Parabola peaking at 30% compression, drops steeply after
perplexity_asr = [0.31, 0.35, 0.31, 0.17, 0.00]
create_mock_data("llama4:scout", "perplexity", "dev_mode", perplexity_asr)
