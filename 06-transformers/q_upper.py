test_upper = Dataset.from_list([make_example(n, num2zh_upper) for n in test_numbers])
print("訓練前 (只學過一般寫法) 大寫數字 accuracy:", evaluate(merged, test_upper))

# 重新從原始模型開始，混合兩種寫法訓練
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
mixed = [make_example(n, rng.choice([num2zh, num2zh_upper])) for n in train_numbers + train_numbers[:2000]]
train_mixed = Dataset.from_list(mixed).shuffle(seed=0).remove_columns("answer")
tokenizer.padding_side = "right"
trainer2 = SFTTrainer(model=model, args=args, train_dataset=train_mixed,
                      processing_class=tokenizer, peft_config=peft_config)
trainer2.train()
print("一般寫法 accuracy:", evaluate(trainer2.model, test_ds, show=0))
print("大寫寫法 accuracy:", evaluate(trainer2.model, test_upper, show=5))
