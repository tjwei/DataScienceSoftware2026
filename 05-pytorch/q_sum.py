n = 10
# 公式
print(n * (n + 1) // 2)
# torch.sum
print(torch.arange(1, n + 1).sum())
# torch.cumsum 的最後一項
print(torch.arange(1, n + 1).cumsum(dim=0)[-1])
# 矩陣乘法 (內積)
print(torch.ones(n) @ torch.arange(1., n + 1))
# python 迴圈
s = torch.tensor(0)
for i in range(1, n + 1):
    s += i
print(s)
