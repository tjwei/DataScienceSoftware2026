import json, re
prompt = """擷取這張收據的資訊，只輸出 JSON，格式：
{"store": str, "date": str, "items": [{"name": str, "qty": int, "price": float}], "subtotal": float, "total": float}"""
answer = ask(receipt, prompt, max_new_tokens=400)
print(answer)
data = json.loads(re.search(r"\{.*\}", answer, re.S).group(0))
items_sum = sum(it["price"] for it in data["items"])
print("品項加總:", round(items_sum, 2), " subtotal:", data["subtotal"], " 相符:", abs(items_sum - data["subtotal"]) < 0.01)
