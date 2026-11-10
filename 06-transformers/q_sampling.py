@torch.no_grad()
def sample(prompt, max_new_tokens=50, temperature=0.7, top_p=0.8):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, use_cache=True)
    generated = []
    for _ in range(max_new_tokens):
        logits = out.logits[0, -1].float() / temperature
        probs = logits.softmax(-1)
        sorted_p, sorted_i = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        keep = cum - sorted_p < top_p          # keep tokens whose cumulative probability *before* adding them is < top_p
        sorted_p = sorted_p * keep
        choice = torch.multinomial(sorted_p / sorted_p.sum(), 1)
        next_id = sorted_i[choice]
        generated.append(next_id.item())
        if next_id.item() == tokenizer.eos_token_id:
            break
        out = model(next_id[None], past_key_values=out.past_key_values, use_cache=True)
    return tokenizer.decode(ids[0].tolist() + generated, skip_special_tokens=True)

for _ in range(3):
    print(sample("The highest mountain in Taiwan is"))
    print("-----")
