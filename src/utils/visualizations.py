import copy

import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from matplotlib.font_manager import findfont, FontProperties


class ImageTextRenderer:
    def __init__(self, size=60):
        font_path = findfont(FontProperties(family='monospace'))
        self.font = ImageFont.truetype(font_path, size=size, index=0)
        self.size = size

    def print_gray(self, img_np_f, text, offs_xy, white=1.0):
        assert len(img_np_f.shape) == 2, "Image must be single channel"
        img_pil = Image.fromarray(img_np_f, mode='F')
        ctx = ImageDraw.Draw(img_pil)
        step = self.size // 15
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ctx.text((offs_xy[0] + step * dx, offs_xy[1] + step * dy), text, font=self.font, fill=0.0)
        ctx.text(offs_xy, text, font=self.font, fill=white)
        return np.array(img_pil)

    def print(self, img_np_f, text, offs_xy, **kwargs):
        if len(img_np_f.shape) == 3:
            for ch in range(3):
                img_np_f[ch] = self.print_gray(img_np_f[ch], text, offs_xy, **kwargs)
        else:
            img_np_f = self.print_gray(img_np_f, text, offs_xy, **kwargs)
        return img_np_f


_text_renderers = dict()


def get_text_renderer(size):
    if size not in _text_renderers:
        _text_renderers[size] = ImageTextRenderer(size)
    return _text_renderers[size]


def img_print(*args, **kwargs):
    size = kwargs['size']
    del kwargs['size']
    renderer = get_text_renderer(size)
    return renderer.print(*args, **kwargs)


def tensor_print(img, caption, **kwargs):
    if isinstance(caption, str) and len(caption.strip()) == 0:
        return img
    assert img.dim() == 4 and img.shape[1] in (1, 3), 'Expecting 4D tensor with RGB or grayscale'
    offset = min(img.shape[2], img.shape[3]) // 100
    img = img.cpu().detach()
    offset = (offset, offset)
    if 'offsetx' in kwargs:
        offset = (kwargs['offsetx'], kwargs['offsety'])
        del kwargs['offsetx'], kwargs['offsety']
    if 'size' in kwargs:
        size = kwargs['size']
        kwargs.pop('size')
    else:
        size = min(img.shape[2], img.shape[3]) // 15
    for i in range(img.shape[0]):
        tag = (caption if isinstance(caption, str) else caption[i]).strip()
        if len(tag) == 0:
            continue
        img_np = img_print(img[i].numpy(), tag, offset, size=size, **kwargs)
        img[i] = torch.from_numpy(img_np)
    return img


def colorize(x, cmap='jet'):
    assert torch.is_tensor(x) and x.dim() == 2 and x.dtype == torch.float32
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    x = cm(x.numpy(), bytes=False)[..., 0:3]
    x = torch.tensor(x).float()
    x = x.permute(2, 0, 1)
    return x


def visualize_singular_values(svs, module_ordering, truncate_singular_values=None, normalize_each=False):
    svs = copy.deepcopy(svs)
    tensor_dims = {name: sv.numel() for name, sv in svs.items()}
    max_tensor_dim = max(tensor_dims.values())
    if truncate_singular_values is not None:
        max_tensor_dim = min(max_tensor_dim, truncate_singular_values)
    num_modules = len(module_ordering)
    canvas = torch.full((max_tensor_dim, num_modules), fill_value=-1.0, dtype=torch.float32)
    sv_min, sv_min_trunc, sv_max = float(np.inf), float(np.inf), -1.0
    for i, name in enumerate(module_ordering):
        if name not in svs:
            continue
        num_svs = tensor_dims[name]
        sv_min = min(sv_min, svs[name].min().item())
        sv_max = max(sv_max, svs[name].max().item())
        if normalize_each:
            svs[name] = svs[name] / sv_max
            sv_min /= sv_max
            sv_max = 1.0
        if truncate_singular_values is not None:
            num_svs = min(num_svs, truncate_singular_values)
            sv_min_trunc = min(sv_min_trunc, svs[name][:num_svs].min().item())
        else:
            sv_min_trunc = sv_min
        canvas[:num_svs, i].copy_(svs[name][:num_svs])
    mask_ignore = canvas < 0
    canvas[mask_ignore] = 0.0
    canvas = (canvas - sv_min_trunc) / max(sv_max - sv_min_trunc, 1e-8)
    canvas = colorize(canvas)
    for i in range(canvas.shape[0]):
        canvas[i][mask_ignore] = 0.
    canvas = F.interpolate(canvas.unsqueeze(0), (max_tensor_dim * 4, num_modules * 4), mode='nearest').squeeze(0)
    canvas = canvas.view(3, -1, 4)
    canvas = torch.cat((canvas, torch.zeros((3, canvas.shape[1], 2), dtype=torch.float32)), dim=2)
    canvas = canvas.reshape(3, max_tensor_dim * 4, num_modules * (4+2))
    margin_top = 16
    margin_right = max(128 - canvas.shape[-1], 0)
    canvas = F.pad(canvas, [0, margin_right, margin_top, 0], value=0.)
    if truncate_singular_values is None or sv_min_trunc == sv_min:
        canvas = tensor_print(canvas.unsqueeze(0), f'{sv_min:.3f} {sv_max:.3f}', size=10).squeeze(0)
    else:
        canvas = tensor_print(
            canvas.unsqueeze(0), f'{sv_min:.3f} {sv_min_trunc:.3f} {sv_max:.3f}', size=10
        ).squeeze(0)
    return canvas
