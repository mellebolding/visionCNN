"""
Distributed training utilities for PyTorch DDP.

Provides clean abstractions for:
- Single GPU training
- Multi-GPU DDP training (local)
- Multi-node DDP training (SLURM/cluster)

Usage:
    # In training script
    from src.utils.distributed import DistributedManager
    
    dist_manager = DistributedManager()
    dist_manager.setup()
    
    model = dist_manager.wrap_model(model)
    train_loader = dist_manager.get_dataloader(train_dataset, ...)
    
    # Only rank 0 should log/save
    if dist_manager.is_main_process:
        logger.info("Training started")
        save_checkpoint(...)
    
    dist_manager.cleanup()
"""
import os
import socket
from typing import Optional
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def get_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_for_distributed(is_main: bool):
    """Disable printing for non-main processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class DistributedManager:
    """
    Manager class for distributed training.
    
    Handles initialization and cleanup of distributed training,
    model wrapping, and dataloader creation with proper samplers.
    
    Attributes:
        rank: Global rank of this process
        local_rank: Local rank on this node
        world_size: Total number of processes
        device: CUDA device for this process
        is_distributed: Whether running in distributed mode
        is_main_process: Whether this is rank 0
    """
    
    def __init__(self, backend: str = "nccl"):
        """
        Initialize the distributed manager.
        
        Args:
            backend: DDP backend ("nccl" for GPU, "gloo" for CPU)
        """
        self.backend = backend
        self._initialized = False
        
        # These will be set in setup()
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = False
        self.is_main_process = True
        
    def setup(self, local_rank: Optional[int] = None):
        """
        Setup distributed training environment.
        
        Automatically detects distributed environment from:
        - torchrun/torch.distributed.launch environment variables
        - SLURM environment variables
        
        Args:
            local_rank: Override local rank (usually from args)
        """
        # Check for distributed environment
        if self._is_slurm_job():
            self._setup_slurm()
        elif self._is_torchrun():
            self._setup_torchrun(local_rank)
        else:
            # Single GPU training
            self._setup_single_gpu()
            return
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size
            )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        self.is_distributed = True
        self.is_main_process = self.rank == 0
        self._initialized = True
        
        # Optionally suppress print for non-main processes
        # setup_for_distributed(self.is_main_process)
        
        # Synchronize before starting
        self.barrier()
        
    def _is_slurm_job(self) -> bool:
        """Check if running in a SLURM job with multiple tasks."""
        return "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1
    
    def _is_torchrun(self) -> bool:
        """Check if launched with torchrun."""
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    def _setup_slurm(self):
        """Setup distributed training from SLURM environment."""
        self.rank = int(os.environ["SLURM_PROCID"])
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        
        # Set master address and port for SLURM
        node_list = os.environ.get("SLURM_NODELIST", "localhost")
        
        # Parse the first node from the SLURM_NODELIST
        if "[" in node_list:
            # Format: node[001-004] -> take first node
            import re
            match = re.match(r"([a-zA-Z]+)\[(\d+)", node_list)
            if match:
                prefix, first_node = match.groups()
                master_addr = f"{prefix}{first_node}"
            else:
                master_addr = node_list.split(",")[0]
        else:
            master_addr = node_list.split(",")[0]
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
    def _setup_torchrun(self, local_rank: Optional[int] = None):
        """Setup distributed training from torchrun environment."""
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        # LOCAL_RANK is set by torchrun
        if local_rank is not None:
            self.local_rank = local_rank
        else:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def _setup_single_gpu(self):
        """Setup for single GPU training."""
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_distributed = False
        self.is_main_process = True
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def wrap_model(
        self, 
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        sync_bn: bool = False
    ) -> torch.nn.Module:
        """
        Wrap model with DDP if distributed, otherwise return as-is.
        
        Args:
            model: Model to wrap
            find_unused_parameters: Set True if some parameters don't contribute to loss
            sync_bn: Convert BatchNorm to SyncBatchNorm for distributed training
            
        Returns:
            Wrapped model (DDP) or original model
        """
        # Move model to device first
        model = model.to(self.device)
        
        if not self.is_distributed:
            return model
        
        # Optionally convert BatchNorm to SyncBatchNorm
        if sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters
        )
        
        return model
    
    def get_sampler(
        self, 
        dataset: Dataset, 
        shuffle: bool = True,
        seed: int = 0
    ) -> Optional[DistributedSampler]:
        """
        Get a DistributedSampler for the dataset if in distributed mode.
        
        Args:
            dataset: Dataset to create sampler for
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
            
        Returns:
            DistributedSampler if distributed, None otherwise
        """
        if not self.is_distributed:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=seed
        )
    
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        seed: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader with proper sampler for distributed training.
        
        Args:
            dataset: Dataset to load
            batch_size: Per-GPU batch size
            shuffle: Whether to shuffle (ignored if distributed, use sampler)
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            seed: Random seed for sampler
            
        Returns:
            DataLoader with appropriate sampler
        """
        sampler = self.get_sampler(dataset, shuffle=shuffle, seed=seed)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def set_epoch(self, sampler: Optional[DistributedSampler], epoch: int):
        """
        Set the epoch for a DistributedSampler.
        
        This must be called at the start of each epoch to ensure proper shuffling.
        
        Args:
            sampler: The DistributedSampler (or None for non-distributed)
            epoch: Current epoch number
        """
        if sampler is not None and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        All-reduce a tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation (SUM, AVG, etc.)
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def reduce_dict(self, input_dict: dict, average: bool = True) -> dict:
        """
        Reduce a dictionary of values across all processes.
        
        Args:
            input_dict: Dictionary with numeric values
            average: Whether to average (True) or sum (False)
            
        Returns:
            Reduced dictionary
        """
        if not self.is_distributed:
            return input_dict
        
        with torch.no_grad():
            keys = sorted(input_dict.keys())
            values = torch.tensor([input_dict[k] for k in keys], device=self.device)
            
            dist.all_reduce(values)
            
            if average:
                values /= self.world_size
            
            return {k: v.item() for k, v in zip(keys, values)}
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
    
    @contextmanager
    def main_process_first(self):
        """
        Context manager to ensure main process runs first.
        
        Useful for downloading datasets or creating directories.
        """
        if not self.is_main_process:
            self.barrier()
        
        yield
        
        if self.is_main_process:
            self.barrier()
    
    def __repr__(self) -> str:
        return (
            f"DistributedManager("
            f"rank={self.rank}, "
            f"local_rank={self.local_rank}, "
            f"world_size={self.world_size}, "
            f"device={self.device}, "
            f"is_distributed={self.is_distributed})"
        )


def get_effective_batch_size(per_gpu_batch_size: int, world_size: int) -> int:
    """Calculate effective batch size across all GPUs."""
    return per_gpu_batch_size * world_size


def scale_learning_rate(base_lr: float, batch_size: int, base_batch_size: int = 256) -> float:
    """
    Scale learning rate based on batch size (linear scaling rule).
    
    Args:
        base_lr: Base learning rate
        batch_size: Effective batch size
        base_batch_size: Reference batch size for base_lr
        
    Returns:
        Scaled learning rate
    """
    return base_lr * batch_size / base_batch_size
