import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7' # which GPUs are visible
os.environ["MASTER_ADDR"] = 'localhost'
os.environ['MASTER_PORT'] = '20001' # address for DDP. one DDP training task corresponds a unique address.

import time
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda import amp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dali_utils import create_dali_pipeline
from utils import get_regular_lr,adjust_lr_iter,save_model,write_txt,setup_seed,AverageMeter,accuracy

from models.darknet53 import darknet53
from models.resnet import resnet50

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except Exception as e:
    print("Please install NVIDIA DALI for boosting training speed.")

# ================================================================================
#                                 All Settings
# ================================================================================
world_size = 8 # How many GPUs is used to DDP
net = "darknet53" # resnet50, darknet53...
imagenet_data_path = "./imagenet1000"
num_classes=1000
train_crop_size = 224 # crop size of training
val_resize_size = 256 # resize size before crop of val
save_dir = "./weights"
start_epoch = 0
end_epoch = 90
total_batch_size = 1024 # total batchsize.
init_lr = total_batch_size*0.1/256 # lr
momentum = 0.9
weight_decay = 1e-4
optimizer = "SGD"
workers = 16 # per gpu workers
lr_step = [30,60] # lr*0.1 at each step
print_interval = 50

seed = 666 # random seed
warmup = True # warmup for first 5 epochs 
amp_use = True # pytorch automatic mixed precision
dali_loader = True # nvidia DALI dataloader
dali_cpu = True # whether to use cpu for DALI. For Large model, "True" is faster for training.
torch.backends.cudnn.benchmark = True
# ================================================================================



# Data loading code
traindir = os.path.join(imagenet_data_path, 'train')
valdir = os.path.join(imagenet_data_path, 'val')


def main_worker(rank,world_size):
    setup_seed(seed + rank)
    #print('rank:',rank,'seed:',seed + rank)
    torch.distributed.init_process_group(backend='nccl',rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)

    best_acc=0
    global start_epoch
    global current_iter

    if net == "resnet50":
        model = resnet50(pretrained=False,num_classes=num_classes).to(rank)
    if net == "darknet53":
        model = darknet53(num_classes=num_classes).to(rank)

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    

    if dali_loader == True:
        # train
        pipe_train = create_dali_pipeline(batch_size=total_batch_size//world_size,
                                    num_threads=workers,
                                    device_id=rank,
                                    seed=seed + rank,
                                    data_dir=traindir,
                                    crop=train_crop_size,
                                    size=val_resize_size,
                                    dali_cpu=dali_cpu,
                                    shard_id=rank,
                                    num_shards=world_size,
                                    is_training=True)
        pipe_train.build()
        train_loader = DALIClassificationIterator(pipe_train, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        # val
        pipe_val = create_dali_pipeline(batch_size=total_batch_size//world_size,
                                    num_threads=workers,
                                    device_id=rank,
                                    seed=seed + rank,
                                    data_dir=valdir,
                                    crop=train_crop_size,
                                    size=val_resize_size,
                                    dali_cpu=dali_cpu,
                                    shard_id=rank,
                                    num_shards=world_size,
                                    is_training=False)
        pipe_val.build()
        val_loader = DALIClassificationIterator(pipe_val, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    else:
        # using pytorch dataloader
        train_dataset = datasets.ImageFolder(
                                            traindir,
                                            transforms.Compose([
                                                                transforms.RandomResizedCrop(train_crop_size),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                                ])
                                            )
        val_dataset = datasets.ImageFolder(
                                            valdir,
                                            transforms.Compose([
                                                                transforms.Resize(val_resize_size),
                                                                transforms.CenterCrop(train_crop_size),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                               ])
                                            )


        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False,drop_last=False)

        train_loader = torch.utils.data.DataLoader(
                                                    train_dataset, 
                                                    batch_size=total_batch_size//world_size, 
                                                    shuffle=False,
                                                    num_workers=workers, 
                                                    pin_memory=True, 
                                                    sampler=train_sampler,
                                                    )
        val_loader = torch.utils.data.DataLoader(
                                                    val_dataset,
                                                    batch_size=total_batch_size//world_size, 
                                                    shuffle=False,
                                                    num_workers=workers, 
                                                    pin_memory=True,
                                                    sampler=val_sampler,
                                                    )

    if torch.cuda.device_count() > 1:
            print("let's use GPU{}, go go go!!!".format(rank))
            model = DDP(model, device_ids=[rank])

    if amp_use == True:
        scaler = amp.GradScaler(enabled=True)

    for epoch in range(start_epoch,end_epoch):
        if dali_loader == False:
            train_sampler.set_epoch(epoch)
        lr_group = get_regular_lr(optimizer)
        start_time = time.time()
        train_epoch(train_loader,model,criterion,optimizer,lr_group,epoch,rank,scaler)
        end_time = time.time()
        ELA_time = (end_time - start_time)
        ELA_time = time.strftime('%H:%M:%S',time.gmtime(ELA_time))
        
      
        # eval
        top1 = validate(val_loader, model,rank)

        # save checkpoint
        if  rank==0:
            if top1 > best_acc:
                best_acc = top1
                save_model(save_dir + '/model_best.pth',epoch,model,optimizer=None)
            save_model(save_dir + '/model_last.pth',epoch,model,optimizer)
            print('Training Time/epoch: ',ELA_time,
                  " | eval best Acc@1:%.4f "%best_acc.item(),
                  " | eval current Acc@1:%.4f "%top1.item())

        if dali_loader == True:
            train_loader.reset()
            val_loader.reset()


def train_epoch(train_loader, model, criterion, optimizer, lr_group, epoch, rank, scaler):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        adjust_lr_iter(optimizer=optimizer, epoch=epoch, lr=init_lr, current_iter=i, len_epoch=len(train_loader),lr_step=lr_step, warmup = warmup)
        # measure data loading time
        if dali_loader == True:
            images = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            images = data[0].cuda(rank,non_blocking=True)
            target = data[1].cuda(rank,non_blocking=True)

        # compute output
        current_base_lr = get_regular_lr(optimizer)[0]

        # update step
        optimizer.zero_grad()
        if amp_use == True:
            with amp.autocast(enabled=True):
                output = model(images)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if rank==0 and i%print_interval==0:
            print('[epoch:{},{}/{}]'.format(epoch,i,len(train_loader)),
                  ' | lr:%.6f '%current_base_lr,
                  ' | loss:%.6f '%loss.item(),
                  ' | training Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


def validate(val_loader, model,rank):

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if dali_loader == True:
                images = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()
            else:     
                images = data[0].cuda(rank,non_blocking=True)
                target = data[1].cuda(rank,non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = reduce_tensor(acc1,world_size) # reduce of acc of all GPUs
            acc5 = reduce_tensor(acc5,world_size)
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg

def reduce_tensor(tensor,world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def main():
    mp.spawn(main_worker,args=(world_size,),nprocs=world_size,join=True)


if __name__ == '__main__':
    main()