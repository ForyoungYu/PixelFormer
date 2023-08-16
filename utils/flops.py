import torch
from thop import clever_format, profile


def flops(model, input_size):
    input = torch.rand(1, 3, input_size, input_size)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops: {}, params: {}".format(flops, params))
