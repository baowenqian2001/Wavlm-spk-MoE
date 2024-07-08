import os

import torch
import torch.distributed as dist

class EarlyStopping:
    """
    Early stops the training if validation loss/metrics doesn't improve after a given patience
    """
    def __init__(self, patience: int = 100, verbose: bool = True, delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 100
            verbose (bool): If True, prints a message for each validation loss improvement. Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss == True:
            score = -metrics
        else:
            score = metrics
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose == True:
                    print(f"EarlyStopping") 
        else:
            self.best_score = score
            self.counter = 0
            
def init_distrubuted_mode(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        opt.rank = int(os.environ['SLURM_PROCID'])
        opt.gpu = opt.rank % torch.cuda.device_count()
    else:
        print('device cannot setup distributed mode')
        opt.distributed = False
        return
    
    opt.distrubuted = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(opt.rank, opt.dist_url), flush=True)

    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
    dist.barrier() #设置一个障碍，即所有rank都运行到这里时才会进行下一步。


def clean_up():
    dist.destroy_process_group()


def is_dist_availble_or_initial():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_availble_or_initial():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_availble_or_initial():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

# 对多个设备上的loss求平均不是为了backward，仅仅是查看做个记录
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value   
    