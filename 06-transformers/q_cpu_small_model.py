small = NeuralCPU(d=256, n_layers=2).to(device)
train(small, TRAIN_STEPS // 2, log_every=2500)
acc_small, per_op_small = exact_rate(small.step, random_batch, N=50_000)
print(f"2-layer model single-step accuracy {acc_small:.4f} (6 layers: {acc_trained:.4f})")
for name in sorted(per_op_small, key=per_op_small.get)[:6]:
    print(f"  {name:5s} 2 layers {per_op_small[name]:.4f}   6 layers {per_op_trained[name]:.4f}")
