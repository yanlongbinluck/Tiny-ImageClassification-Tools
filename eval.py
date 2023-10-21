
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from models.darknet53 import darknet53
from models.resnet import resnet18,resnet34,resnet50,resnet101,resnet152


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



def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            #print(i)
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(num_classes,5)))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    return top1.avg

if __name__ == '__main__':

    valdir = os.path.join("/home/data/yanlb/dataset/imagenet", 'val')
    net = "resnet50"
    num_classes = 1000

    
    if net == "resnet18":
        model = resnet18(pretrained=False,num_classes=num_classes)
    if net == "resnet34":
        model = resnet34(pretrained=False,num_classes=num_classes)
    if net == "resnet50":
        model = resnet50(pretrained=False,num_classes=num_classes)
    if net == "resnet101":
        model = resnet101(pretrained=False,num_classes=num_classes)
    if net == "resnet152":
        model = resnet152(pretrained=False,num_classes=num_classes)
    if net == "darknet53":
        model = darknet53(num_classes=num_classes)

    checkpoint = torch.load("./work_dir/" + net + "/resnet50_75.93.pth")
    model.load_state_dict(checkpoint)



    model.cuda()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=256, 
        shuffle=False,
        num_workers=16, 
        pin_memory=True)
    top1 = validate(val_loader, model)
    print("top1 acc is: {:.2f}%".format(top1.cpu().numpy()))
