test_upper = Dataset.from_list([make_example(n, num2zh_upper) for n in test_numbers])
print("before (trained on ordinary numerals only), formal-numeral accuracy:", evaluate(merged, test_upper))

# Start again from the original model and train on a mix of both forms
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
mixed = [make_example(n, rng.choice([num2zh, num2zh_upper])) for n in train_numbers + train_numbers[:2000]]
train_mixed = Dataset.from_list(mixed).shuffle(seed=0).remove_columns("answer")
tokenizer.padding_side = "right"
trainer2 = SFTTrainer(model=model, args=args, train_dataset=train_mixed,
                      processing_class=tokenizer, peft_config=peft_config)
trainer2.train()
print("ordinary numerals accuracy:", evaluate(trainer2.model, test_ds, show=0))
print("formal numerals accuracy:", evaluate(trainer2.model, test_upper, show=5))
