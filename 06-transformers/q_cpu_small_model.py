small = NeuralCPU(d=256, n_layers=2).to(device)
train(small, TRAIN_STEPS, log_every=2000)
acc_small, per_op_small = evaluate(small, 50000)
print(f"2 層模型單步正確率 {acc_small:.4f} (6 層: {acc:.4f})")
for name in sorted(per_op_small, key=per_op_small.get)[:6]:
    print(f"  {name:5s} 2層 {per_op_small[name]:.4f}   6層 {per_op[name]:.4f}")
