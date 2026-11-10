top_p, top_i = torch.topk(probs, 5)
for p, i in zip(top_p.tolist(), top_i.tolist()):
    print(f"{weights.meta['categories'][i]:30s} {p:.3f}")
