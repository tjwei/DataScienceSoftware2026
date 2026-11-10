"""中文數字 <-> 阿拉伯數字 的資料產生工具 (06-transformers/05 與 HW 使用)"""
import random

DIGITS = "零一二三四五六七八九"
UPPER = dict(zip("零一二三四五六七八九十百千萬", "零壹貳參肆伍陸柒捌玖拾佰仟萬"))


def _section(n):
    """0 < n < 10000"""
    s, zero = "", False
    for unit, p in (("千", 1000), ("百", 100), ("十", 10), ("", 1)):
        d = n // p % 10
        if d == 0:
            zero = bool(s)
        else:
            if zero:
                s += "零"
                zero = False
            s += DIGITS[d] + unit
    return s


def num2zh(n):
    """0 <= n < 10**12, 例如 10050 -> 一萬零五十"""
    if n == 0:
        return "零"
    s, skipped = "", False
    for val, unit in ((n // 10**8, "億"), (n // 10**4 % 10**4, "萬"), (n % 10**4, "")):
        if val:
            if s and (val < 1000 or skipped):
                s += "零"
            s += _section(val) + unit
            skipped = False
        elif s:
            skipped = True
    if s.startswith("一十"):
        s = s[1:]
    return s


def num2zh_upper(n):
    """大寫 (支票寫法), 例如 3052 -> 參仟零伍拾貳"""
    s = num2zh(n)
    if s.startswith("十"):
        s = "一" + s
    return "".join(UPPER.get(c, c) for c in s)


def random_number(rng=random):
    """位數平均分布的隨機整數"""
    k = rng.randint(1, 8)
    n = rng.randint(10 ** (k - 1), 10**k - 1)
    if rng.random() < 0.3:  # 多一些含有 0 的數字
        n = int("".join(c if rng.random() < 0.6 else "0" for c in str(n)) or 0)
    return n
