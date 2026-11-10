def my_softmax(z):
    z = z - z.max(dim=1, keepdim=True).values   # numerical stability: subtract the max first
    e = z.exp()
    return e / e.sum(dim=1, keepdim=True)

def my_cross_entropy(logits, y):
    p = my_softmax(logits)
    return -p[torch.arange(len(y)), y].log().mean()

def my_cross_entropy2(logits, y):
    # log softmax = z - logsumexp(z)
    log_p = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_p[torch.arange(len(y)), y].mean()

logits = torch.randn(5, 10) * 3
y = torch.randint(0, 10, (5,))
print(my_softmax(logits).sum(dim=1))
print(my_cross_entropy(logits, y), my_cross_entropy2(logits, y), F.cross_entropy(logits, y))
