"""
Inspired by https://arxiv.org/pdf/1908.10999.pdf.
This implementation follows the "dynamic compensation" path, but instead of relying on the external
spectral normalization, as done by the authors in the official neurips20 repository
(https://github.com/max-liu-112/SRGANs/tree/18379994b2fed12a47004db66dbc43e63ea6a0bb/dis_models),
this implementation rewrites singular values in-place, so the learned parameters do not grow in magnitude.
"""
import torch


def spectral_compensation_stateful(module, state=None, classes=None, normalize=False, truncation=None, eps=1e-7):
    if classes is None:
        raise ValueError('Classes subjected to spectral regularization are not specified')
    if state is None:
        state = {}
    for n, m in module.named_modules():
        if not any(isinstance(m, a) for a in classes):
            continue
        shape = m.weight.shape
        if isinstance(m, torch.nn.modules.conv._ConvTransposeNd):
            mat = m.weight.detach().permute(1, 0, *range(2, len(shape))).reshape(shape[1], -1)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) \
                or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Embedding):
            mat = m.weight.detach().view(shape[0], -1)
        else:
            raise NotImplementedError(f'Class {type(m)} dispatch not implemented')
        old_sv = state.get(n, None)
        if old_sv is None:
            _, s, _ = mat.svd(compute_uv=False)
            if truncation is not None and s.numel() > truncation:
                s = s[:truncation]
            s /= s.max().clamp_min(eps)
            state[n] = s
            continue
        u, s, v = mat.svd()
        if truncation is not None and s.numel() > truncation:
            u = u[:, :truncation]
            s = s[:truncation]
            v = v[:, :truncation]
        s_max = s.max().clamp_min(eps)
        s /= s_max
        state[n] = torch.max(s, old_sv)
        s = state[n]
        if not normalize:
            s *= s_max
        mat = u.mm(s.view(-1, 1) * v.T)
        if isinstance(m, torch.nn.modules.conv._ConvTransposeNd):
            mat = mat.reshape(shape[1], shape[0], *shape[2:]).permute(1, 0, *range(2, len(shape))).reshape(*shape)
        else:
            mat = mat.reshape(*shape)
        with torch.no_grad():
            m.weight.copy_(mat)
    return state
