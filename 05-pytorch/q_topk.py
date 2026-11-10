@torch.no_grad()
def generate_topk(model, idx, max_new_tokens, temperature=1.0, top_k=10, block_size=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if block_size is None else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")    # 比第 k 大還小的都設為 -inf
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx

with torch.autocast(device, dtype=torch.float16):
    print(decode(generate_topk(model, start, 300, temperature=1.0, top_k=5, block_size=block_size)[0].tolist()))
