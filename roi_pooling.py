import numpy as np
import torch
import torch.autograd as ag
from torch.autograd.function import Function
from torch._thnn import type2backend


class AdaptiveMaxPool2d(torch.autograd.Function):
    def __init__(self, out_w, out_h):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0], size[1])(input)


def roi_pooling_imgs(input, rois, size=(7,7), spatial_scale=1.0):
    assert(rois.dim() == 2)
    assert(rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    
    rois[:,1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)

if __name__ == '__main__':
    input = ag.Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
    rois = ag.Variable(torch.LongTensor([[1, 2, 7, 8], [3, 3, 8, 8]]), requires_grad=False)

    out = roi_pooling_ims(input, rois, size=(8, 8))
    out.backward(out.data.clone().uniform_())
