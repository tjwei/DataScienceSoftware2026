for kwargs in [dict(amp=False), dict(amp=True), dict(amp=True, compile=True)]:
    print(kwargs)
    fit(resnet9(), train_gpu_loader, epochs=2, **kwargs)
