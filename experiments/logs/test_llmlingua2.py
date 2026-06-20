from llmlingua import PromptCompressor
compressor = PromptCompressor("gpt2", use_llmlingua2=False, device_map="cpu")
res = compressor.compress_prompt(["Hello world, this is a very long test prompt."], rate=0.5, force_tokens=["."])
print(type(res))
if isinstance(res, dict):
  print(res.keys())
  print(res)
