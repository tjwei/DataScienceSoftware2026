# 06 🤗 Transformers 與 LLM

所有 notebook 都可以直接在 Google Colab 上執行：打開後選 `執行階段 → 變更執行階段類型 → T4 GPU`。
建議先完成 [05-pytorch](../05-pytorch)。

| Notebook | 內容 | Colab |
|---|---|---|
| [01-Pipelines](01-Pipelines.ipynb) | `pipeline`：情緒分析、zero-shot、NER、填空、LLM 生成、Whisper 語音辨識、SigLIP 2、深度估計 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/01-Pipelines.ipynb) |
| [02-Tokenizer-Model-Generate](02-Tokenizer-Model-Generate.ipynb) | BPE tokenizer、logits、自己寫生成迴圈、KV cache、sampling、chat template、思考模式、perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/02-Tokenizer-Model-Generate.ipynb) |
| [03-Finetune-Text-Classification](03-Finetune-Text-Classification.ipynb) | `datasets` + `Trainer` 微調 mmBERT 做中文情緒分類 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/03-Finetune-Text-Classification.ipynb) |
| [04-LLM-Apps](04-LLM-Apps.ipynb) | 多輪對話、批次生成、JSON 結構化輸出、工具呼叫 (agent)、RAG (Qwen3-Embedding)、4-bit 量化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/04-LLM-Apps.ipynb) |
| [05-LoRA-SFT](05-LoRA-SFT.ipynb) | LoRA 原理、TRL `SFTTrainer` + PEFT 微調 Qwen3-0.6B、評估、合併權重 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/05-LoRA-SFT.ipynb) |
| [06-Vision-Language](06-Vision-Language.ipynb) | SigLIP 2 以文搜圖與 zero-shot 分類、Qwen3-VL 看圖說話、OCR、資訊擷取 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/06-Vision-Language.ipynb) |
| [HW1-Sentiment](HW1-Sentiment.ipynb) | 作業：中文情緒分類準確率挑戰 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/HW1-Sentiment.ipynb) |
| [HW2-LoRA-Check-Writer](HW2-LoRA-Check-Writer.ipynb) | 作業：用 LoRA 教 Qwen3-0.6B 把數字寫成支票大寫 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/06-transformers/HW2-LoRA-Check-Writer.ipynb) |

## 使用的模型（都可以在 T4 上執行，不需要申請授權）

| 模型 | 用途 |
|---|---|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)、[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (4-bit) | LLM、思考模式、工具呼叫、LoRA 微調 |
| [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 多模態 LLM |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | RAG 文字向量 |
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 多語言 encoder (ModernBERT 架構) |
| [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | 影像-文字向量 |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 語音辨識 |

## 練習 (Q) 與參考解答

notebook 中標示 `## Q` 的是練習題，參考解答在 `q_*.py`，Colab 上會自動下載。
* `%run -i q_xxx.py`：執行解答
* `!cat q_xxx.py`：看解答程式碼

`zh_numbers.py` 是 05-LoRA-SFT 和 HW2 用來產生中文數字資料的工具。

## T4 GPU 注意事項

* T4 不支援 bf16：載入模型用 `dtype=torch.float16`，訓練用 `fp16=True, bf16=False`（TRL 的 `SFTConfig` 預設會開 bf16）
* 不支援 FlashAttention 2：使用預設的 SDPA attention 即可
* 記憶體 15GB：換模型前用 `del model; gc.collect(); torch.cuda.empty_cache()` 釋放記憶體，或重新啟動執行階段
* 測試環境：PyTorch 2.10、transformers 5.22、TRL 1.17、PEFT 0.21
