import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, datas, category = None): # [1, 3, 224, 224]

        image_size = (datas.size(-1), datas.size(-2))
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0) # [1, 3, 224, 224]

            # for resnet
            for name, module in self.model.named_children():
                if name == 'fc':
                    feature = feature.view(feature.size(0), -1) # [1, 512, 7, 7] to [1, 25088]
                feature = module(feature)
                if name == 'layer4':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature

            # for VGGnet
            # for name, module in self.model.named_children():
            #     if name == 'classifier':
            #         feature = feature.view(feature.size(0), -1) # [1, 512, 7, 7] to [1, 25088]
            #     feature = module(feature)
            #     if name == 'features':
            #         feature.register_hook(self.save_gradient)
            #         self.feature = feature

            classes = torch.sigmoid(feature) # torch.Size([1, 1000])
            if category is None:
                one_hot, _ = classes.max(dim=-1) # tensor([0.9998], grad_fn=<MaxBackward0>)
            else:
                one_hot = classes[0][category]

            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            del self.gradient


            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            #cam = heat_map
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))


        heat_maps = torch.stack(heat_maps)
        return heat_maps
