"""
Cliente LLM con dos modos:
- local: ejecuta modelos de Hugging Face en la máquina
- api: usa endpoint compatible OpenAI
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline import config


class LLMClient:
    def __init__(self, model: str | None = None, backend: str | None = None):
        self.backend = (backend or config.BACKEND).lower()
        self.model = model or config.LLM_MODEL
        self.client = None
        self.tokenizer = None
        self.local_model = None

        if self.backend == "api":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.API_KEY,
                base_url=config.API_BASE_URL,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

    def generate(self, prompt: str, max_tokens: int = 512) -> dict:
        """Envía un prompt al LLM y devuelve respuesta + metadatos."""
        if self.backend == "api":
            return self._generate_api(prompt, max_tokens)
        return self._generate_local(prompt, max_tokens)

    def _generate_api(self, prompt: str, max_tokens: int) -> dict:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        latency = time.time() - start
        content = response.choices[0].message.content or ""
        usage = response.usage

        return {
            "response": content,
            "latency": latency,
            "tokens_prompt": usage.prompt_tokens if usage else 0,
            "tokens_completion": usage.completion_tokens if usage else 0,
        }

    def _generate_local(self, prompt: str, max_tokens: int) -> dict:
        start = time.time()

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"System: {config.SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"

        model_device = self.local_model.device
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_len:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency = time.time() - start

        return {
            "response": content,
            "latency": latency,
            "tokens_prompt": int(input_len),
            "tokens_completion": int(generated_ids.shape[0]),
        }
