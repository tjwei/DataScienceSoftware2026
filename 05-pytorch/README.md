# 05 PyTorch

Every notebook runs on Google Colab: open it and choose `Runtime → Change runtime type → T4 GPU`.

| Notebook | Topics | Colab |
|---|---|---|
| [01-PyTorch-Basics](01-PyTorch-Basics.ipynb) | Tensors, GPU, fp16 speed, autograd, gradient descent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/01-PyTorch-Basics.ipynb) |
| [02-From-PyTorch-to-MNIST](02-From-PyTorch-to-MNIST.ipynb) | Logistic regression from scratch → `nn.Module` → MLP → CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/02-From-PyTorch-to-MNIST.ipynb) |
| [03-Modern-Training-Loop](03-Modern-Training-Loop.ipynb) | CIFAR10, Dataset/DataLoader, GPU data augmentation, ResNet, mixed precision (fp16), `torch.compile`, OneCycle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/03-Modern-Training-Loop.ipynb) |
| [04-Transfer-Learning](04-Transfer-Learning.ipynb) | torchvision weights API, fine-tuning a timm ConvNeXt, DINOv2 features + logistic regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/04-Transfer-Learning.ipynb) |
| [05-Attention-and-GPT](05-Attention-and-GPT.ipynb) | Self-attention from scratch, SDPA, an RMSNorm Transformer, training a character-level GPT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/05-Attention-and-GPT.ipynb) |
| [HW1-CIFAR10](HW1-CIFAR10.ipynb) | Homework: best CIFAR10 accuracy within 10 minutes of training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjwei/DataScienceSoftware2026/blob/main/05-pytorch/HW1-CIFAR10.ipynb) |

## Exercises (Q) and reference solutions

Sections marked `## Q` are exercises; reference solutions are in `q_*.py`.
On Colab, the first code cell of each notebook downloads the solution files automatically.
* `%run -i q_xxx.py` runs a solution
* `!cat q_xxx.py` shows its code (in Jupyter you can also use `%load q_xxx.py` to load it into the cell)

## Notes for the T4 GPU

* The T4 (Turing, compute capability 7.5) has fp16 Tensor Cores but **no native bf16 and no FlashAttention 2**, so training uses `torch.autocast(dtype=torch.float16)` + `GradScaler`.
* `F.scaled_dot_product_attention` automatically uses a memory-efficient attention kernel that the T4 supports.
* Colab has only 2 CPU cores; for small datasets, keep the whole dataset on the GPU and do data augmentation there too.
* Measured on a Colab T4: ResNet9 reaches about 91% on CIFAR10 in 10 epochs (~90 s); fp16 mixed precision is 3–4× faster than fp32, while `torch.compile` does not speed up this small model.
* Tested with PyTorch 2.10 and torchvision 0.25 (the Colab defaults in 2026).
