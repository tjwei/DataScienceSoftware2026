# 05 PyTorch

所有 notebook 都可以直接在 Google Colab 上執行：打開後選 `執行階段 → 變更執行階段類型 → T4 GPU`。

| Notebook | 內容 | Colab |
|---|---|---|
| [01-PyTorch-Basics](01-PyTorch-Basics.ipynb) | Tensor、GPU、fp16 速度、autograd、gradient descent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/01-PyTorch-Basics.ipynb) |
| [02-From-PyTorch-to-MNIST](02-From-PyTorch-to-MNIST.ipynb) | 從零寫 logistic regression → `nn.Module` → MLP → CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/02-From-PyTorch-to-MNIST.ipynb) |
| [03-Modern-Training-Loop](03-Modern-Training-Loop.ipynb) | CIFAR10、Dataset/DataLoader、GPU 資料增強、ResNet、AMP (fp16)、`torch.compile`、OneCycle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/03-Modern-Training-Loop.ipynb) |
| [04-Transfer-Learning](04-Transfer-Learning.ipynb) | torchvision weights API、timm ConvNeXt 微調、DINOv2 特徵 + logistic regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/04-Transfer-Learning.ipynb) |
| [05-Attention-and-GPT](05-Attention-and-GPT.ipynb) | 從零寫 self-attention、SDPA、RMSNorm Transformer，訓練字元層級 GPT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/05-Attention-and-GPT.ipynb) |
| [HW1-CIFAR10](HW1-CIFAR10.ipynb) | 作業：10 分鐘內 CIFAR10 準確率挑戰 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/HW1-CIFAR10.ipynb) |

## 練習 (Q) 與參考解答

notebook 中標示 `## Q` 的是練習題，參考解答在 `q_*.py`。
在 Colab 上，每個 notebook 的第一個程式格會自動下載解答檔。
* `%run -i q_xxx.py`：執行解答
* `!cat q_xxx.py`：看解答程式碼（Jupyter 也可以用 `%load q_xxx.py` 載入到格子裡）

## T4 GPU 注意事項

* T4 (Turing, compute capability 7.5) 有 fp16 Tensor Core，但**不支援原生 bf16 與 FlashAttention 2**，所以訓練都用 `torch.autocast(dtype=torch.float16)` + `GradScaler`
* `F.scaled_dot_product_attention` 會自動使用 T4 支援的 memory-efficient attention kernel
* Colab 只有 2 個 CPU 核心，小資料集建議整個放在 GPU 上，連資料增強一起在 GPU 做
* 測試環境：PyTorch 2.10、torchvision 0.25（與 2026 年 Colab 預設相同）
