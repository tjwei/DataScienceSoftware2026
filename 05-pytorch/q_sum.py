n = 10
# formula
print(n * (n + 1) // 2)
# torch.sum
print(torch.arange(1, n + 1).sum())
# last element of torch.cumsum
print(torch.arange(1, n + 1).cumsum(dim=0)[-1])
# matrix multiplication (dot product)
print(torch.ones(n) @ torch.arange(1., n + 1))
# python loop
s = torch.tensor(0)
for i in range(1, n + 1):
    s += i
print(s)
