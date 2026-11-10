class ShardedStream(IterableDataset):
    def __init__(self, n=10):
        self.n = n
    def __iter__(self):
        info = get_worker_info()
        if info is None:                      # no workers: the main process yields everything
            start, step = 0, 1
        else:                                 # worker k of W takes k, k+W, k+2W, ...
            start, step = info.id, info.num_workers
        for i in range(start, self.n, step):
            yield i, worker_id()

loader = DataLoader(ShardedStream(), batch_size=4, num_workers=NUM_WORKERS)
seen = []
for numbers, workers in loader:
    print("numbers", numbers.tolist(), " produced by worker", workers.tolist())
    seen += numbers.tolist()
print("every number exactly once:", sorted(seen) == list(range(10)))
# The DataLoader takes batches from the workers in turn (worker 0, worker 1, worker 0, ...),
# so the batches interleave the two shards, and each batch comes from a single worker.
