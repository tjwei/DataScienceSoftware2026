import copy

def noisy_copy(model, sigma):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * sigma)
    return m

for sigma in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
    acc, per_op = exact_rate(noisy_copy(designed, sigma).step, random_batch, N=40_000)
    worst = sorted(per_op, key=per_op.get)[:4]
    print(f"sigma={sigma:g}: single-step exact accuracy {acc:.4f}   worst: "
          + ", ".join(f"{name} {per_op[name]:.2f}" for name in worst))

# Why it is fragile:
# * the gate ReLU(v - 255 (1 - c)) multiplies the boolean by 255, so a tiny error in c becomes a big error;
# * the lookup tables add and subtract many large ReLU values that must cancel exactly;
# * the attention scale (1000) multiplies every error in the query and key.
# A trained network spreads information over many dimensions and is usually smoother.
