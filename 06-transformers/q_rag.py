from urllib.request import urlopen, Request
from urllib.parse import quote
# Use the plain text of the Wikipedia article "Taiwan" as an example (via the Wikipedia API)
title = "Taiwan"
url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles="
       + quote(title))
data = json.loads(urlopen(Request(url, headers={"User-Agent": "course-example/1.0"})).read())
text = next(iter(data["query"]["pages"].values()))["extract"]
docs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
print(len(docs), "paragraphs")
doc_emb = embedder.encode(docs, batch_size=32, show_progress_bar=True)

for q in ["What is the highest mountain in Taiwan?", "Roughly how large is Taiwan's area?", "What are the indigenous peoples of Taiwan?"]:
    print(f"👤 {q}\n🤖 {rag(q)}\n")
