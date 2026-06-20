# Resultados Finales — Paper IEEE

Resultados del pipeline agrupados por **paradigma de compresión** y **modelo objetivo**.

## Estructura

```
results_final/
├── llmlingua2/              # Compresión extractiva (LLMLingua-2)
│   ├── llama4_scout/        # Meta-Llama-4-Scout 109B (MoE)
│   ├── qwen_110b/           # Qwen-1.5-110B-Chat (Dense)
│   └── gemma2_27b/          # Gemma-2-27B-it (Dense)
└── bart/                    # Compresión abstractiva (BART-large-cnn)
    ├── llama4_scout/
    ├── qwen_110b/
    └── gemma2_27b/
```

Cada archivo sigue el formato: `{template}__experiment_{YYMMDD}_{HHMMSS}.json`

Templates:
- `raw` — prompt malicioso directo
- `dev_mode` — Developer Mode (jailbreak)
- `aim` — Always Intelligent and Machiavellian (jailbreak)
- `dan` — Do Anything Now (jailbreak)

Estos 24 archivos (3 modelos × 4 templates × 2 compressores) son los que corresponden
al experimento final descrito en el paper, con el juez Llama-Guard-3-8B y ~1200 muestras
por archivo. La tasa de compresión base (control) está incluida dentro de cada JSON.
