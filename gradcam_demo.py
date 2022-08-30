# based on: https://github.com/leftthomas/GradCAM

import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import glob
import json
import cv2
import numpy as np
import torch
from torchvision import models
from gradcam_utils import GradCam
from PIL import Image
from torchvision import transforms
from models.resnet import resnet50

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        #print(path+' --> has been created')
def get_keys(d,value):
    return [k for k ,v in d.items() if v == value]


if __name__ == '__main__':
    
    image_path = './input_image'
    grad_val = './gradcam_image'
    json_path = './imagenet_class_index.json'


    # make dir
    sub_path = [x for x in os.listdir(image_path)]
    for i in range(len(sub_path)):
        mkdir(grad_val+'/'+sub_path[i])

    if True:
        model = resnet50(pretrained=False,num_classes=1000)
        checkpoint = torch.load("./weights/resnet50_75.93.pth")
        model.load_state_dict(checkpoint)
        model.cuda()
    else:
        model = models.resnet50(pretrained=True).cuda()

    grad_cam = GradCam(model)

    # category index
    with open(json_path,'r') as f:
        category_dict = json.load(f)


    Transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    for i in range(len(sub_path)):
        full_dir = image_path + '/' + sub_path[i]
        index = [k for k,v in category_dict.items() if str(sub_path[i]) in v]
        print(i,index,sub_path[i])
        targets = int(index[0])
        img_list = sorted(glob.glob(full_dir+'/*.*'))

        for j in range(len(img_list)):
            filename_without_dir = os.path.basename(img_list[j])
            img = Image.open(img_list[j])
            if (img.mode!='RGB'):
                img = img.convert('RGB')
            test_image = Transforms(img).unsqueeze(dim=0).cuda()

            feature_image = grad_cam(test_image,category = targets).squeeze(dim=0) 
            feature_image = transforms.ToPILImage()(feature_image)
            feature_image.save(grad_val+'/'+sub_path[i] + "/{}".format(filename_without_dir))
            torch.cuda.empty_cache()



