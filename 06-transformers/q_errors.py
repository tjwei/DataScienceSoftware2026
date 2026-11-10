pred = trainer.predict(ds_tok["test"])
y_pred = pred.predictions.argmax(-1)
y_true = pred.label_ids
wrong = np.where(y_pred != y_true)[0]
print(f"錯了 {len(wrong)} / {len(y_true)} 筆")
for i in wrong[:10]:
    print(f"[答案 {id2label[y_true[i]]} 預測 {id2label[y_pred[i]]}] {ds['test'][int(i)]['text'][:120]}")
