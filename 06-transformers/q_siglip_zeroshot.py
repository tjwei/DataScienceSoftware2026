texts = [f"a photo of a {name.replace('_', ' ')}, a type of pet." for name in class_names]
txt_emb = text_embeddings(texts)
pred = (img_emb @ txt_emb.T).argmax(1).cpu().numpy()
print("zero-shot accuracy:", (pred == labels).mean())
for i in range(5):
    print(f"predicted {class_names[pred[i]]:28s} answer {class_names[labels[i]]}")
