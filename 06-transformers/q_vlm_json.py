import json, re
prompt = """Extract the information from this receipt. Output only JSON, in this format:
{"store": str, "date": str, "items": [{"name": str, "qty": int, "price": float}], "subtotal": float, "total": float}"""
answer = ask(receipt, prompt, max_new_tokens=400)
print(answer)
data = json.loads(re.search(r"\{.*\}", answer, re.S).group(0))
items_sum = sum(it["price"] for it in data["items"])
print("Items sum:", round(items_sum, 2), " subtotal:", data["subtotal"], " match:", abs(items_sum - data["subtotal"]) < 0.01)
