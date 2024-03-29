

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import time
import json
import glob
import cv2
import numpy as np
from models.darknet53 import darknet53
from models.resnet import resnet18,resnet34,resnet50,resnet101,resnet152

if __name__ == '__main__':
    
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
    model.eval()

    with open('./imagenet_class_index.json', 'r') as f:
        load_dict = json.load(f)

    img_list = sorted(glob.glob('/home/data/yanlb/dataset/imagenet/val/n01682714/*.*'))
    
    for i in range(len(img_list)):
        inp = cv2.imread(img_list[i]) / 255. # [H,W,3]
        inp = cv2.resize(inp, (224,224))
        inp = inp[:,:,::-1] # BGR2RGB
        mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(1, 1, 3) # RGB format
        std  = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(1, 1, 3) # RGB format
        inp = (inp - mean) / std    
        inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(dim=0).float()
        out = model(inp.cuda()).cpu().detach()
        value,index = torch.max(out,dim=1)
        print("predicted class name in imagenet--->",load_dict[str(index.item())]," | ","predicted index--->", index)

