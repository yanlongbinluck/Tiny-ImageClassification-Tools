from thop import clever_format,profile
import torch
from models.darknet53 import darknet53
from models.resnet import resnet50

net = "resnet50"

inp = torch.randn((1,3,224,224),requires_grad=False)

if net == "resnet50":
    model = resnet50(pretrained=False,num_classes=1000)
if net == "darknet53":
    model = darknet53(num_classes=1000)


macs,params = profile(model,inputs=(inp,))
macs,params = clever_format([macs,params],"%.3f")
print("FLOPs:",macs,"Params:",params) # actually, the units of computation is macs, but in paper, people usually name it FLOPs.
