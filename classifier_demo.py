

import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import time
import json
import glob
import cv2
import numpy as np
from models.darknet53 import darknet53
from models.resnet import resnet50

if __name__ == '__main__':
    
    net = "resnet50"
    if net == "resnet50":
        model = resnet50(pretrained=False,num_classes=1000)
    if net == "darknet53":
        model = darknet53(num_classes=1000)

    checkpoint = torch.load("./weights/resnet50_75.93.pth")
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    

    with open('./imagenet_class_index.json', 'r') as f:
        load_dict = json.load(f)

    img_list = sorted(glob.glob('./input_image/n01682714/*.*'))
    
    for i in range(len(img_list)):
        inp = cv2.imread(img_list[i]) / 255. # [H,W,3]
        inp = cv2.resize(inp, (224,224)) # size可以随意改变，不影响分类结果
        mean = np.array([0.406, 0.456, 0.485],dtype=np.float32).reshape(1, 1, 3)
        std  = np.array([0.225, 0.224, 0.229],dtype=np.float32).reshape(1, 1, 3)
        inp = (inp - mean) / std

        inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(dim=0).float()
        out = model(inp.cuda()).cpu().detach()
        value,index = torch.max(out,dim=1)
        print("predicted results--->",load_dict[str(index.item())],"predicted index--->", index)

