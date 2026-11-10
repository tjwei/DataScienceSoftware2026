for s in ["今天天氣很好，我們去公園散步吧。",
          "The weather is nice today, let's take a walk in the park.",
          "1234567", "😀🎉👍", "def f(x):\n    return x**2",
          "臺灣", "台灣"]:
    ids = tokenizer.encode(s)
    print(len(ids), [tokenizer.decode([i]) for i in ids])
