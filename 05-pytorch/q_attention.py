def my_attention(q, k, v):
    T, d = q.size(-2), q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)              # (B, h, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, float("-inf"))             # can't look at the future
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights

B, h, T, d = 2, 4, 8, 16
q, k, v = torch.randn(3, B, h, T, d, device=device).unbind(0)
y1, weights = my_attention(q, k, v)
y2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("max diff:", (y1 - y2).abs().max().item())
print(weights[0, 0])
