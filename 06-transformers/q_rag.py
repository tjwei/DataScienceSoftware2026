from urllib.request import urlopen, Request
from urllib.parse import quote
# 以維基百科「臺灣」條目的純文字為例 (使用 Wikipedia API)
title = "臺灣"
url = ("https://zh.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&variant=zh-tw&titles="
       + quote(title))
data = json.loads(urlopen(Request(url, headers={"User-Agent": "course-example/1.0"})).read())
text = next(iter(data["query"]["pages"].values()))["extract"]
docs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
print(len(docs), "段")
doc_emb = embedder.encode(docs, batch_size=32, show_progress_bar=True)

for q in ["臺灣最高的山是什麼？", "臺灣的面積大約多大？", "臺灣有哪些原住民族？"]:
    print(f"👤 {q}\n🤖 {rag(q)}\n")
