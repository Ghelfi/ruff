from torch.utils.data import DataLoader, DistributedSampler

dataset = []
DataLoader(dataset, sampler=DistributedSampler(dataset))  # noqa: TORCH300
DataLoader(dataset, sampler=DistributedSampler(dataset))
