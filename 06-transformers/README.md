# 06 🤗 Transformers and LLMs

Every notebook runs on Google Colab: open it and choose `Runtime → Change runtime type → T4 GPU`.
We recommend finishing [05-pytorch](../05-pytorch) first.

| Notebook | Topics | Colab |
|---|---|---|
| [01-Pipelines](01-Pipelines.ipynb) | `pipeline`: sentiment analysis, zero-shot classification, NER, fill-mask, LLM generation, Whisper speech recognition, SigLIP 2, depth estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/01-Pipelines.ipynb) |
| [02-Tokenizer-Model-Generate](02-Tokenizer-Model-Generate.ipynb) | BPE tokenizers, logits, writing your own generation loop, KV cache, sampling, chat templates, thinking mode, perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/02-Tokenizer-Model-Generate.ipynb) |
| [03-Finetune-Text-Classification](03-Finetune-Text-Classification.ipynb) | Fine-tuning mmBERT for sentiment classification with `datasets` + `Trainer` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/03-Finetune-Text-Classification.ipynb) |
| [04-LLM-Apps](04-LLM-Apps.ipynb) | Multi-turn chat, batch generation, structured JSON output, tool calling (agents), RAG with Qwen3-Embedding, 4-bit quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/04-LLM-Apps.ipynb) |
| [05-LoRA-SFT](05-LoRA-SFT.ipynb) | How LoRA works, fine-tuning Qwen3-0.6B with TRL `SFTTrainer` + PEFT, evaluation, merging weights | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/05-LoRA-SFT.ipynb) |
| [06-Vision-Language](06-Vision-Language.ipynb) | SigLIP 2 text-to-image search and zero-shot classification; Qwen3-VL image description, OCR, information extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/06-Vision-Language.ipynb) |
| [A1-Neural-CPU](A1-Neural-CPU.ipynb) | Extra: a Transformer that *is* a tiny 8-bit CPU — weights designed by hand (exact, Tracr-style) vs. trained from examples; accuracy, error accumulation, robustness, and whether it could run an NES | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/A1-Neural-CPU.ipynb) |
| [A2-Transformer-Turing-Machine](A2-Transformer-Turing-Machine.ipynb) | Extra: a hand-designed Transformer that runs **any** (small) Turing machine given its transition table as input: attention as exact lookup, ReLU logic gates, one forward pass per step; busy beaver; is a Transformer Turing complete? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/A2-Transformer-Turing-Machine.ipynb) |

Some examples use Chinese text on purpose (Chinese NER, Chinese reviews, Chinese numerals) to show multilingual models and a task that small LLMs genuinely get wrong before fine-tuning.

## Models used (all run on a T4 and need no access request)

| Model | Used for |
|---|---|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (4-bit) | LLM, thinking mode, tool calling, LoRA fine-tuning |
| [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Vision-language model |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Text embeddings for RAG |
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | Multilingual encoder (ModernBERT architecture) |
| [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | Image–text embeddings |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Speech recognition |

## Exercises (Q) and reference solutions

Sections marked `## Q` are exercises; reference solutions are in `q_*.py` and are downloaded automatically on Colab.
* `%run -i q_xxx.py` runs a solution
* `!cat q_xxx.py` shows its code

`zh_numbers.py` generates the Chinese-numeral data used by 05-LoRA-SFT.

## Notes for the T4 GPU

* The T4 has no bf16: load models with `dtype=torch.float16` and train with `fp16=True, bf16=False` (TRL's `SFTConfig` enables bf16 by default).
* No FlashAttention 2: the default SDPA attention works fine.
* 15 GB of GPU memory: free it before loading another model with `del model; gc.collect(); torch.cuda.empty_cache()`, or restart the runtime.
* Colab's preinstalled `torchao` is too old for recent PEFT; the LoRA notebooks uninstall it in their first cell.
* Tested on a Colab T4 in November 2026 with PyTorch 2.14, transformers 5.22, TRL 1.17 and PEFT 0.21.
