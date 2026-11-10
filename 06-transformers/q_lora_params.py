total = sum(p.numel() for p in trainer.model.parameters())
trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
n_trainable = sum(p.numel() for _, p in trainable)
print(f"trainable params {n_trainable:,} / total {total:,} = {n_trainable / total:.2%}")
for n, p in trainable[:6]:
    print(n, tuple(p.shape), p.dtype)
