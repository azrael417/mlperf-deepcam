import os
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
    if torch.cuda.is_available():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = 0
        
    return local_rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def init(method):
    #get master address and port
    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        #use that URI
        address = addrport.split(":")[0]
        #use the default pytorch port
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE",0))
        
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
        
    elif method == "nccl-slurm":
        rank = int(os.getenv("PMIX_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)

    elif method == "nccl-slurm-pmi":
        rank = int(os.getenv("PMI_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
                                                
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
        
    elif method == "mpi":
        #init DDP
        dist.init_process_group(backend = "mpi")
        
    else:
        raise NotImplementedError()
