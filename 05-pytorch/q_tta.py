@torch.no_grad()
def evaluate_tta(model):
    model.eval()
    correct = 0
    for xb, yb in batches(test_X, test_y, 1000):
        xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device, dtype=torch.float16):
            logits = model(xb).float().softmax(1) + model(xb.flip(3)).float().softmax(1)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(test_X)

print("without TTA:", evaluate(model))
print("with TTA   :", evaluate_tta(model))
