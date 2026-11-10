import time
def finetune(model_id, epochs=2, lr=5e-5, batch_size=32):
    tok = AutoTokenizer.from_pretrained(model_id)
    data = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True, remove_columns=["text"])
    m = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, id2label=id2label, label2id=label2id)
    a = TrainingArguments(output_dir="tmp-" + model_id.split("/")[-1], per_device_train_batch_size=batch_size,
                          per_device_eval_batch_size=64, learning_rate=lr, num_train_epochs=epochs, warmup_steps=0.1,
                          fp16=torch.cuda.is_available(), eval_strategy="no", save_strategy="no",
                          logging_steps=100, report_to="none")
    t = Trainer(model=m, args=a, train_dataset=data["train"], processing_class=tok, compute_metrics=compute_metrics)
    t0 = time.perf_counter()
    t.train()
    dt = time.perf_counter() - t0
    acc = t.evaluate(data["test"])["eval_accuracy"]
    print(f"{model_id}: test acc={acc:.4f}, training time {dt:.0f}s")
    del m, t
    torch.cuda.empty_cache()

finetune("google-bert/bert-base-chinese")
