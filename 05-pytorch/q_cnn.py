def cnn_model():
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 10),
    )
model = cnn_model()
train(model, epochs=5)
