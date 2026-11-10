# 物件偵測 (object detection) 範例
from PIL import ImageDraw
detector = pipeline("object-detection", model="PekingU/rtdetr_v2_r18vd", device=device)
results = detector(img, threshold=0.5)
im = img.copy()
draw = ImageDraw.Draw(im)
for r in results:
    b = r["box"]
    draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline="red", width=4)
    draw.text((b["xmin"] + 5, b["ymin"] + 5), f"{r['label']} {r['score']:.2f}", fill="red")
display(im.resize((400, int(400 * im.height / im.width))))
results
