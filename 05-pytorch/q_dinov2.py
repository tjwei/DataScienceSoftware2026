from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

@torch.no_grad()
def extract(split, batch_size=64):
    feats, labels = [], []
    ds = pets[split]
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i:i + batch_size]
        inputs = processor(images=[im.convert("RGB") for im in batch["image"]], return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16):
            out = dino(**inputs)
        feats.append(out.pooler_output.float().cpu())
        labels += batch["label"]
    return torch.cat(feats).numpy(), np.array(labels)

train_f, train_l = extract("train")
test_f, test_l = extract("test")
print(train_f.shape)
clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(train_f, train_l)
print("DINOv2-small + LogisticRegression test acc:", clf.score(test_f, test_l))
