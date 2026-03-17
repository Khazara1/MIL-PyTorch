import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, Subset
import torch.distributed as dist
from typing import List


# Given a batch of bags with varying number of instances, collate them into a single tensor with padding. Returns features, labels, masks, and bags length.
def collate_fn(batch):
    # Aquires important dimensions
    batch_size = len(batch)
    c, h, w = batch[0][0].shape[1:]
    max_bag_length = max([len(x) for x, y, _, _ in batch])

    instances_idx = [instance_idx for _, _, instance_idx, _ in batch]
    instances_cords = [instance_cords for _, _, _, instance_cords in batch]
    
    # Initializing placeholders for features and labels
    features = torch.zeros((batch_size*max_bag_length, c, h, w))
    labels = torch.zeros((len(batch)), dtype=torch.long)

    # Masking placeholder, mask = 1 for valid instances, 0 for padded instances
    masks = torch.zeros((batch_size*max_bag_length))

    # Empty image used for padding
    pad_image = torch.zeros((1, c, h, w))

    for i, (x, y, _, _) in enumerate(batch):
        n_instances, c, h, w = x.shape

        # Set features and labels
        features[i*max_bag_length:(i*max_bag_length+n_instances)] = x
        features[(i*max_bag_length+n_instances):(i+1)*max_bag_length] = pad_image.expand((max_bag_length-n_instances, c, h, w))

        masks[i*max_bag_length:(i*max_bag_length+n_instances)] = 1
        labels[i] = y

    return features, labels, masks, max_bag_length, instances_idx, instances_cords


def create_dataloader(dataset, batch_size, num_workers, is_ddp, rank=0, world_size=1, sample_type=None, shuffle=True, seed=42):
    """
    Create a DataLoader for the given dataset, handling both distributed and non-distributed settings.

    Batch size is the batch size used per rank in distributed mode.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    targets = dataset.labels # Get all targets from the dataset

    if not is_ddp:
        if shuffle:
            # Use weighted random sampler to handle class imbalance
            counts = torch.unique(targets, return_counts=True) # Count occurrences of each class
            
            # Calculate weights for each class
            weights = 1.0 / counts[1].float()
            samples_weights = weights[targets]

            # Create sampler
            weighted_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
            
            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, sampler=weighted_sampler, pin_memory=True, generator=g)

            return dataloader, weighted_sampler
        else:
            return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=False, pin_memory=True, generator=g), None
    else:
        if rank == 0:
            all_idx = balance_indices(targets, sample_type=sample_type)
            all_idx = torch.tensor(all_idx, dtype=torch.int64)
        else:
            all_idx = None
        
        # Create object list to hold all indices and broadcast to all ranks
        # broadcast_object_list is used for Python objects (lists, numpy arrays, etc.)
        obj_list = [all_idx]

        # Broadcast all indices to all ranks
        dist.broadcast_object_list(obj_list, src=0)
        if isinstance(obj_list[0], torch.Tensor):
            all_idx = obj_list[0].tolist()
        
        subset = Subset(dataset, all_idx)

        sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle)

        dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, sampler=sampler, pin_memory=True, generator=g)

        return dataloader, sampler

def balance_indices(targets: torch.Tensor, sample_type="oversample") -> List:
    """
    Given a tensor of targets, return balanced indices for sampling.
    """
    assert sample_type in ["oversample", "undersample", None], "sample_type must be either 'oversample', 'undersample' or None"

    counts = torch.unique(targets, return_counts=True) # Count occurrences of each class

    if sample_type == "oversample":
        # Calculate weights for each class
        weights = 1.0 / counts[1].float()
        samples_weights = weights[targets]

        # Create sampler
        weighted_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

        return list(weighted_sampler)
    elif sample_type == "undersample":
        # Get the minimum count among all classes
        min_count = torch.min(counts[1]).item()

        balanced_indices = []

        for class_idx in counts[0]:
            # Get all indices for the current class
            class_indices = torch.where(targets == class_idx)[0]

            # Randomly select min_count indices from the current class
            selected_indices = class_indices[torch.randperm(len(class_indices))[:min_count]]

            # Add selected indices to the balanced list
            balanced_indices.extend(selected_indices.tolist())

        balanced_indices = torch.tensor(balanced_indices)
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]  # Shuffle

        return balanced_indices.tolist()
    else:
        return torch.arange(len(targets)).tolist()