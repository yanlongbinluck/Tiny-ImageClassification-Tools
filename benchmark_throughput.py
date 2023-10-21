import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import time
from models.darknet53 import darknet53
from models.resnet import resnet50

net = "resnet50"
if __name__ == '__main__':
    batch = 32
    warmup_iter = 100 # for GPU warmup
    length = 100 # total times of measurement
    inp = torch.randn((batch,3,224,224),requires_grad=False)

    if net == "resnet50":
        model = resnet50(pretrained=False,num_classes=1000)
    if net == "darknet53":
        model = darknet53(num_classes=1000)

    model.eval()
    if True:
        model = model.cuda()
        inp = inp.cuda()
    

    total_time = 0
    with torch.no_grad():
        for i in range(length+warmup_iter):
            print(i)
            torch.cuda.synchronize()
            start = time.time()
            _ = model(inp)
            torch.cuda.synchronize()
            end = time.time()
            if i >= warmup_iter:
                total_time = total_time + (end - start)
    per_time = total_time/(batch*length)
    print('Throughput -->', 1/per_time)
