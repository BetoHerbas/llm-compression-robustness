from llmlingua import PromptCompressor
import warnings
warnings.filterwarnings('ignore')
compressor = PromptCompressor("gpt2", use_llmlingua2=False, device_map="cpu")
res = compressor.compress_prompt(["Hello world, this is a very long test prompt that we want to compress heavily. Let us see how it works."], ratio=0.5)
print("KEYS:", res.keys())
print("RES:", res)
