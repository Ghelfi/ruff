"""Test cases for TORCH303: `DataLoader` workers without `worker_init_fn`."""

from torch.utils.data import DataLoader

dataset = []


def seed_worker(worker_id):
    pass


# Errors (should trigger TORCH303)
DataLoader(dataset, num_workers=4)
DataLoader(dataset, num_workers=1, batch_size=32)

# Should NOT trigger TORCH303
DataLoader(dataset, num_workers=0)
DataLoader(dataset)  # Defaults to num_workers=0.
DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)
DataLoader(dataset, num_workers=workers)  # Non-literal: don't guess.
