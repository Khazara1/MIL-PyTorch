import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


# Deactrivates batchnorm layers to avoid issues related to different bag size in Multiple Instance Learning
def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.track_running_stats = False
        model.running_mean = None
        model.running_var = None


def build_model(model_class, your_model_args, is_ddp, rank, local_rank, state_dict=None, is_mil=False, device="cuda" if torch.cuda.is_available() else "cpu"):
    if is_ddp:
        if rank == 0:
            model = model_class(**your_model_args)
            if is_mil:
                model.apply(deactivate_batchnorm)

            if state_dict is None:
                sd = model.state_dict()
            else:
                sd = state_dict
        else:
            model = model_class(**your_model_args)

            if is_mil:
                model.apply(deactivate_batchnorm)
            sd = None

        obj_list = [sd]
        # broadcast model state dict
        dist.broadcast_object_list(obj_list, src=0)
        sd = obj_list[0]
        model.load_state_dict(sd)
    else:
        model = model_class(**your_model_args)

        if is_mil:
            model.apply(deactivate_batchnorm)

        model.load_state_dict(state_dict)
    
    model = model.to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    return model