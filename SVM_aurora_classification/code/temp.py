import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, inception_v3
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

model = shufflenet_v2_x1_0(weights='DEFAULT')
model2 = inception_v3(weights='DEFAULT')
a = get_graph_node_names(model=model)
b = get_graph_node_names(model=model2)
a0 = a[0]
a1 = a[1]
b1 = b[1]
b0 = b[0]
print(len(b0))
print(len(b1))
print(len(a0))
print(len(a1))