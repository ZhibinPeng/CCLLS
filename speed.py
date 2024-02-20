import torch
from torchvision.models import resnet50, resnet18
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import model_fs

model = resnet18(num_classes=7)
# model = model_fs.ResNet18_Scale()

tensor = (torch.rand(1, 3, 224, 224),)

flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())

print(parameter_count_table(model))
