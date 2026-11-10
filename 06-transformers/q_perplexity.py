@torch.no_grad()
def perplexity(text):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    loss = model(ids, labels=ids).loss
    return loss.exp().item()

for s in ["今天天氣很好，我們去公園散步吧。",
          "今天公園很好，我們去天氣散步吧。",
          "天今很好氣，我公散們去步園吧。",
          "The weather is nice today, let's go for a walk in the park.",
          "The park is nice today, let's go for a walk in the weather.",
          "weather The nice is, today walk go let's a for park the in."]:
    print(f"{perplexity(s):10.2f}  {s}")
