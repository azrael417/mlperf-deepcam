import torch
import torch.distributed as dist

def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    
    #number of GPUs per node
    n = torch.cuda.device_count() // dist.get_size()
    
    return dist.get_rank() % n


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_size()
    else:
        size = 1
    return size
