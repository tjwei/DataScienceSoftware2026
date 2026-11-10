import matplotlib.pyplot as plt
x = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True)
y = torch.sin(x)
# y is a vector; sum it into a scalar before backward, so each element's gradient is d sin(x_i)/dx_i
y.sum().backward()
plt.plot(x.detach(), y.detach(), label="sin(x)")
plt.plot(x.detach(), x.grad, label="autograd d/dx sin(x)", linewidth=4, alpha=0.5)
plt.plot(x.detach(), torch.cos(x.detach()), "--", label="cos(x)")
plt.legend()
plt.show()
print("max error:", (x.grad - torch.cos(x.detach())).abs().max().item())
