"""
Prompt Guard (meta-llama/Prompt-Guard-86M).
Clasificador local que detecta BENIGN / INJECTION / JAILBREAK.
Requiere aceptar la licencia en https://huggingface.co/meta-llama/Prompt-Guard-86M
y configurar `huggingface-cli login` si el modelo es gated.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline import config


class PromptGuard:
    LABELS = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}

    def __init__(self, model_name: str | None = None):
        model_name = model_name or config.PROMPT_GUARD_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def detect(self, text: str) -> dict:
        """Clasifica un texto como BENIGN, INJECTION o JAILBREAK."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

        return {
            "label": self.LABELS[pred_idx],
            "score": round(probs[pred_idx].item(), 4),
            "scores": {
                self.LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(self.LABELS))
            },
        }
