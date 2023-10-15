import torch
import numpy as np
import random
import os

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def get_regular_lr(optimizer):
    lr_group=[] # get  all lr for every group
    for param_group in optimizer.param_groups:
        lr_group += [param_group['lr']]
    return lr_group


def adjust_lr_iter(optimizer, epoch, lr, current_iter, len_epoch, lr_step, warmup):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = sum(i < epoch + 1 for i in lr_step)
    lr = lr*(0.1**factor)

    """Warmup"""
    if warmup == True:
        if epoch < 5:
            lr = lr*float(1 + current_iter + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(path,epoch,model,optimizer = None, save_best = False, save_to_old_version = False):
    if isinstance(model,torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model,torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {'epoch':epoch,'state_dict':state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    
    if save_best == False:
        torch.save(state_dict,path)
    else:
        if save_to_old_version == False:
            torch.save(model.state_dict(),path, _use_new_zipfile_serialization=True)
        else:
            torch.save(model.state_dict(),path, _use_new_zipfile_serialization=False) # this is for pytorch with old version, such as pytorch1.0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def write_txt(txt_file,str_input,write_mode = "a"):
    with open(txt_file,write_mode) as f:
        f.writelines([str_input, '\n'])