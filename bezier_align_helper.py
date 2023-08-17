from adet.layers import BezierAlign
from PIL import Image, ImageOps
import numpy as np

import os.path as osp
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, scale=1.25):
        super(Model, self).__init__()
        self.bezier_align = BezierAlign(output_size, scale, 1)
        self.masks = nn.Parameter(torch.ones(input_size, dtype=torch.float32))

    def forward(self, input, rois):
        # apply mask
        x = input * self.masks
        rois = self.convert_to_roi_format(rois)
        return self.bezier_align(x, rois)

    def convert_to_roi_format(self, beziers):
        concat_boxes = cat([b for b in beziers], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(beziers)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

def get_size(image_size, w, h):
    w_ratio = w / image_size[1]
    h_ratio = h / image_size[0]
    down_scale = max(w_ratio, h_ratio)
    if down_scale > 1:
        return down_scale
    else:
        return 1

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def bezier_align_main(im, cps, res_path, scale=1, device=torch.device('cuda')):
    image_size = (2560, 2560)  # H x W
    output_size = (256, 1024)

    input_size = (image_size[0] // scale,
                  image_size[1] // scale)
    m = Model(input_size, output_size, 1 / scale).to(device)

    beziers = [[]]
    im_arrs = []
    down_scales = []
    
   
    # pad
    w, h = im.size
    down_scale = get_size(image_size, w, h)
    down_scales.append(down_scale)
    if down_scale > 1:
        im = im.resize((int(w / down_scale), int(h / down_scale)), Image.ANTIALIAS)
        w, h = im.size
    padding = (0, 0, image_size[1] - w, image_size[0] - h)
    im = ImageOps.expand(im, padding)
    im = im.resize((input_size[1], input_size[0]), Image.ANTIALIAS)
    im_arrs.append(np.array(im))


    beziers[0].append(cps)

    beziers = [torch.from_numpy(np.stack(b)).to(device).float() for b in beziers]
    beziers = [b / d for b, d in zip(beziers, down_scales)]

    im_arrs = np.stack(im_arrs)
    x = torch.from_numpy(im_arrs).permute(0, 3, 1, 2).to(device).float()

    x = m(x, beziers)
    for i, roi in enumerate(x):
        roi = roi.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(roi, "RGB")
        im.save(f'{res_path}')

def bezier_align_crops(im, cps, res_path, scale=1, device=torch.device('cuda'), return_tensor = True):
    image_size = (2560, 2560)  # H x W
    output_size = (256, 1024)

    input_size = (int(image_size[0] // scale),
                  int(image_size[1] // scale))
    m = Model(input_size, output_size, 1 / scale).to(device)

    beziers = [[]]
    im_arrs = []
    down_scales = []
    
   
    # pad
    w, h = im.size
    down_scale = get_size(image_size, w, h)
    down_scales.append(down_scale)
    if down_scale > 1:
        im = im.resize((int(w / down_scale), int(h / down_scale)), Image.ANTIALIAS)
        w, h = im.size
    padding = (0, 0, image_size[1] - w, image_size[0] - h)
    im = ImageOps.expand(im, padding)
    im = im.resize((input_size[1], input_size[0]), Image.ANTIALIAS)
    im_arrs.append(np.array(im))



    beziers = [torch.from_numpy(cps).to(device).float() for b in beziers]
    beziers = [b / d for b, d in zip(beziers, down_scales)]

    im_arrs = np.stack(im_arrs)
    x = torch.from_numpy(im_arrs).permute(0, 3, 1, 2).to(device).float()

    x = m(x, beziers)
    if return_tensor:
        return x 
    for i, roi in enumerate(x):
        roi = roi.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(roi, "RGB")
        im.save(osp.join(res_path, f'{i}.png'))