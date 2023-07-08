import torch

from biformer.models.biformer import biformer_tiny, biformer_base, biformer_small

# test
model_tiny = biformer_tiny(pretrained=False)
model_base = biformer_base(pretrained=False)
model_small = biformer_small(pretrained=False)

input = torch.randn(2, 3, 224, 224)
output = model_small(input)
for i in output:
    print(i.shape)

