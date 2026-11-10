def rosenbrock(p):
    x, y = p
    return (1 - x)**2 + 100 * (y - x**2)**2

for opt_name, opt_cls, lr in [("SGD", torch.optim.SGD, 1e-3), ("Adam", torch.optim.Adam, 1e-2)]:
    p = torch.tensor([-1.0, 2.0], requires_grad=True)
    optimizer = opt_cls([p], lr=lr)
    for i in range(5001):
        optimizer.zero_grad()
        loss = rosenbrock(p)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"{opt_name} step {i:5d} loss={loss.item():.6f} p={p.detach().numpy()}")
