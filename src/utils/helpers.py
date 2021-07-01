import contextlib
import copy
import difflib
import json
import os
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from functools import partial

import numpy as np
import packaging.version
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch_fidelity.metric_fid import KEY_METRIC_FID
from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD
from torch_fidelity.metric_ppl import KEY_METRIC_PPL_MEAN


def deep_transform(model, callback, prefix=None, opaque=None):
    if not issubclass(type(model), torch.nn.Module):
        return
    for k, v in list(model.named_children()):
        full_prefix = k if prefix is None else prefix + '.' + k
        setattr(model, k, callback(v, full_prefix, opaque))
        deep_transform(v, callback, full_prefix, opaque)


def is_item_ignored(n, m):
    if n.endswith('num_batches_tracked'):
        # Useless BatchNorm stats
        assert m.dtype == torch.int64
        return True
    return False


def get_statedict_num_params(sd):
    assert type(sd) is OrderedDict and all(torch.is_tensor(v) for v in sd.values())
    sd = {k: v for k, v in sd.items() if not is_item_ignored(k, v)}
    assert all(v.dtype == torch.float32 for v in sd.values())
    return sum([torch.tensor(m.shape).prod().item() for m in sd.values()])


def dict_deep_get(d, key_path, split_ch='/', default=None, create_if_missing=False, dict_type=dict):
    if type(key_path) is str:
        parts = key_path.split(split_ch)
    elif type(key_path) is list:
        parts = key_path
    else:
        assert False
    for i, part in enumerate(parts):
        is_last = (i == len(parts)-1)
        if part in d:
            d = d[part]
        else:
            if create_if_missing:
                if is_last:
                    d[part] = default
                else:
                    d[part] = dict_type()
                d = d[part]
            else:
                return default
    return d


def dict_flatten(d, prefix='', join_char='/'):
    out = {}
    for k, v in d.items():
        cur_k = k if prefix == '' else prefix + join_char + k
        if isinstance(v, dict):
            out.update(dict_flatten(v, cur_k, join_char))
        elif isinstance(v, list) or isinstance(v, tuple):
            out.update(dict_flatten({f'{i}': a for i, a in enumerate(v)}, cur_k, join_char))
        else:
            out[cur_k] = v
    return out


def module_deep_get(d, key_path, split_ch='.'):
    if type(key_path) is str:
        parts = key_path.split(split_ch)
    elif type(key_path) is list:
        parts = key_path
    else:
        assert False
    for i, part in enumerate(parts):
        d = getattr(d, part)
        if d is None:
            return None
    return d


def torch_save_atomic(what, path):
    path_tmp = path + '.tmp'
    torch.save(what, path_tmp)
    os.rename(path_tmp, path)


def state_dict_size(state_dict):
    size = 0

    def _state_dict_size(sd):
        nonlocal size
        if type(sd) is torch.Tensor:
            print('Found tensor shape', sd.shape, list(sd.__dict__.keys()))
            size += torch.tensor(sd.shape).prod().item()
        elif type(sd) in (OrderedDict, dict):
            for v in sd.values():
                _state_dict_size(v)

    _state_dict_size(state_dict)
    return size


def get_model_size_bytes(model):
    f = None
    try:
        f = tempfile.NamedTemporaryFile()
        torch.save(model.state_dict(), f)
        return os.path.getsize(f.name)
    finally:
        if f is not None:
            f.close()


def net_extract_modules_order(net, dummy_input, classes_interest, net_prefix=None, classes_ignored=None):
    ignored_prefixes = []

    def cb_hook(prefix, out_module_order, module, input, output):
        out_module_order.append(prefix)

    def cb_embed_tracing_hook(module, prefix, opaque):
        if classes_ignored is not None:
            if type(module) in classes_ignored:
                ignored_prefixes.append(prefix)
                return module
            if any(prefix.startswith(bp) for bp in ignored_prefixes):
                return module
        if type(module) in classes_interest:
            opaque['hooks'].append(module.register_forward_hook(partial(cb_hook, prefix, opaque['module_order'])))
        return module

    out = {'hooks': [], 'module_order': []}

    deep_transform(net, cb_embed_tracing_hook, prefix=net_prefix, opaque=out)

    with torch.no_grad():
        net(*dummy_input)

    for hook in out['hooks']:
        hook.remove()

    return out['module_order']


class PersistentRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_train_total, rng_seed=2020):
        self.dataset = dataset
        self.num_samples_train_total = num_samples_train_total
        self.next_sample_id = 0
        self.sample_ids = []

        rng = np.random.RandomState(rng_seed)

        while len(self.sample_ids) < num_samples_train_total:
            self.sample_ids += rng.permutation(len(dataset)).tolist()

        self.sample_ids = self.sample_ids[:num_samples_train_total]

    def __iter__(self):
        return iter(self.sample_ids[self.next_sample_id:])

    def __len__(self):
        return self.num_samples_train_total

    def state_dict(self):
        # not saving next_sample_id because sampler is used in look-ahead fashion by DataLoader
        return {key: value for key, value in self.__dict__.items() if key != 'dataset' and key != 'next_sample_id'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def fast_forward_to(self, sample_id):
        self.next_sample_id = sample_id


class ModuleEMA(torch.nn.Module):
    def __init__(self, src, placement='same', momentum=0.9999):
        super().__init__()
        assert placement in ('same', 'cpu')
        self.src = [src]
        self.placement = placement
        self.momentum = momentum
        self.dst = copy.deepcopy(src)
        self.dst.zero_grad()
        self.dst.eval()
        if placement == 'cpu':
            self.dst = self.dst.cpu()

    def update(self):
        with torch.no_grad():
            for prop in ('named_parameters', 'named_buffers'):
                for (ns, ps), (nd, pd) in zip(getattr(self.src[0], prop)(), getattr(self.dst, prop)()):
                    assert ns == nd, f'{prop} mismatch: ns="{ns}" and nd="{nd}"'
                    if ns.endswith('.num_batches_tracked'):
                        continue
                    assert hasattr(ps, 'data') and hasattr(pd, 'data'), f'{nd} has no .data'
                    assert torch.is_floating_point(ps.data) and torch.is_floating_point(pd.data), f'{nd} not a float'
                    ps_data = ps.data
                    if self.placement == 'cpu':
                        ps_data = ps_data.cpu()
                    pd.data *= self.momentum
                    pd.data += ps_data * (1 - self.momentum)

    def forward(self, *args, cuda=True, **kwargs):
        dst = self.dst
        if cuda and self.placement == 'cpu':
            dst = self.dst.cuda()
        with torch.no_grad():
            return dst.forward(*args, **kwargs)


def generate_validation_fakes(g_model, num_samples, batch_size, z_sz, num_classes, root_out, rng_seed=2020):
    os.makedirs(root_out, exist_ok=False)
    num_samples_filename_digits = int(np.log10(num_samples)) + 1
    rng = np.random.RandomState(rng_seed)
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(num_samples, batch_start + batch_size)
            batch_seed = rng.randint(2 ** 32)
            fake_z = generate_noise(
                batch_end - batch_start, z_sz, num_classes=num_classes, rng_seed=batch_seed, device=g_model
            )
            fake_rgb = g_model.forward(*fake_z)
            fake_rgb = (255 * (fake_rgb.clamp(-1, 1) * 0.5 + 0.5)).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            for i in range(batch_end - batch_start):
                path = os.path.join(root_out, f'{i + batch_start:0{num_samples_filename_digits}d}.png')
                assert not os.path.isfile(path)
                img = Image.fromarray(fake_rgb[i], mode='RGB')
                img.save(path)


def add_filetree_to_zip(zip, dir_src, filter_filename=None, filter_dirname=None):
    dir_src = os.path.abspath(dir_src)
    dir_src_name = os.path.basename(dir_src)
    dir_src_parent_dir = os.path.dirname(dir_src)
    zip.write(dir_src, arcname=dir_src_name)
    for cur_dir, _, cur_filenames in os.walk(dir_src):
        if filter_dirname is not None and filter_dirname(os.path.basename(cur_dir)):
            continue
        if cur_dir != dir_src:
            zip.write(cur_dir, arcname=os.path.relpath(cur_dir, dir_src_parent_dir))
        for filename in cur_filenames:
            if filter_filename is not None and filter_filename(filename):
                continue
            zip.write(
                os.path.join(cur_dir, filename),
                arcname=os.path.join(os.path.relpath(cur_dir, dir_src_parent_dir), filename)
            )


def pack_source_and_configuration(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    cfg = copy.deepcopy(cfg.__dict__)
    del cfg['log_dir']
    cfg_str = json.dumps(cfg, indent=4)
    with zipfile.ZipFile(path_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(
            zip,
            dir_src,
            filter_filename=lambda f: not f.endswith('.py'),
            filter_dirname=lambda d: d in ('__pycache__',),
        )
        zip.writestr('cfg.txt', cfg_str)


def pack_directory(path_dir, path_zip, filter_filename):
    path_dir = os.path.abspath(path_dir)
    with zipfile.ZipFile(path_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(zip, path_dir, filter_filename=filter_filename)


def diff_source_dir_and_zip(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    with zipfile.ZipFile(path_zip) as zip:
        for file in zip.namelist():
            if file == 'cfg.txt':
                continue
            file_info = zip.getinfo(file)
            if file_info.is_dir():
                continue
            path_src = os.path.join(os.path.dirname(dir_src), file)
            if not os.path.isfile(path_src):
                raise FileNotFoundError(path_src)
            with open(path_src) as f:
                lines_src = f.read().split('\n')
            lines_zip = zip.read(file).decode('utf-8').split('\n')
            lines_diff = list(difflib.unified_diff(lines_zip, lines_src))
            if len(lines_diff) > 0:
                raise Exception(
                    f'Source ({file}) changed - will not resume. Diff:\n' +
                    f'\n'.join(lines_diff)
                )
        cfg = copy.deepcopy(cfg.__dict__)
        del cfg['log_dir']
        cfg_str = json.dumps(cfg, indent=4).split('\n')
        cfg_zip = zip.read('cfg.txt').decode('utf-8').split('\n')
        cfg_diff = list(difflib.unified_diff(cfg_zip, cfg_str))
        if len(cfg_diff) > 0:
            raise Exception(
                f'Configuration changed - will not resume. Diff:\n' +
                f'\n'.join(cfg_diff)
            )


def verify_experiment_integrity(cfg):
    dir_src = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    path_zip = os.path.join(cfg.log_dir, 'source.zip')
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.isfile(path_zip):
        pack_source_and_configuration(cfg, dir_src, path_zip)
    else:
        diff_source_dir_and_zip(cfg, dir_src, path_zip)


def get_singular_values_from_network(net, classes_interest, classes_ignored=None):
    svs = {}
    ignored_prefixes = []
    for n, m in net.named_modules():
        if classes_ignored is not None:
            if type(m) in classes_ignored:
                ignored_prefixes.append(n)
                continue
            if any(n.startswith(bp) for bp in ignored_prefixes):
                continue
        if type(m) not in classes_interest:
            continue
        assert hasattr(m, 'weight'), f'{n} has no weight'
        name = 'weight'
        if hasattr(m, 'weight_orig'):
            # use trainable parameters if the standard torch.nn.utils.spectral_norm is attached
            name = 'weight_orig'
        weight = getattr(m, name).detach()
        conv_transposed = isinstance(m, torch.nn.modules.conv._ConvNd) and is_conv_transposed(m)
        if conv_transposed:
            weight = weight.permute(1, 0, *range(2, len(weight.shape)))
        weight = weight.reshape(weight.shape[0], -1)
        sv = weight.svd(compute_uv=False).S
        svs[n] = sv
    return svs


def get_best_gan_metrics(current, last=None):
    if last is None:
        return current
    out = last
    for k, v_cur in current.items():
        if k == KEY_METRIC_ISC_MEAN:
            if v_cur > out[k]:
                out[KEY_METRIC_ISC_MEAN] = current[KEY_METRIC_ISC_MEAN]
                out[KEY_METRIC_ISC_STD] = current[KEY_METRIC_ISC_STD]
        elif k == KEY_METRIC_FID:
            if v_cur < out[k]:
                out[KEY_METRIC_FID] = current[KEY_METRIC_FID]
        elif k == KEY_METRIC_KID_MEAN:
            if v_cur < out[k]:
                out[KEY_METRIC_KID_MEAN] = current[KEY_METRIC_KID_MEAN]
                out[KEY_METRIC_KID_STD] = current[KEY_METRIC_KID_STD]
        elif k == KEY_METRIC_PPL_MEAN:
            if v_cur < out[k]:
                out[KEY_METRIC_PPL_MEAN] = current[KEY_METRIC_PPL_MEAN]
    return out


def get_best_imgcls_metrics(current, last=None):
    if last is None:
        return current
    out = last
    for k, v_cur in current.items():
        if v_cur > out.get(k, 0):
            out[k] = v_cur
    return out


def tb_add_scalars(tb, main_tag, tag_scalar_dict, global_step=None):
    # unlike SummaryWriter.add_scalars, this function does not create a separate FileWriter per each dict entry
    for k, v in tag_scalar_dict.items():
        tag = main_tag + '/' + k
        if isinstance(v, dict):
            tb_add_scalars(tb, tag, v, global_step=global_step)
        else:
            tb.add_scalar(tag, v, global_step=global_step)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stderr_redirected(to=os.devnull, stderr=None):
    # https://stackoverflow.com/a/22434262/411907
    if stderr is None:
       stderr = sys.stderr

    stderr_fd = fileno(stderr)
    # copy stderr_fd before it is overwritten; `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        stderr.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stderr_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stderr_fd)  # $ exec > to
        try:
            yield stderr  # allow code to be run with the redirected stdout
        finally:
            # restore stderr to its previous value; dup2 makes stderr_fd inheritable unconditionally
            stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)  # $ exec >&copied


class SilentSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        with stderr_redirected():
            super().__init__(*args, **kwargs)


def silent_torch_jit_trace_module(*args, **kwargs):
    with warnings.catch_warnings(record=True) as ws:
        out = torch.jit.trace_module(*args, **kwargs)
        for w in ws:
            if issubclass(w.category, torch.jit.TracerWarning):
                with open(w.filename) as fp:
                    lines = fp.readlines()
                line = lines[w.lineno-1]
                if '# produces TracerWarning -- safe to ignore' in line:
                    continue
            print(w)
        return out


def is_conv_transposed(conv):
    assert isinstance(conv, torch.nn.modules.conv._ConvNd)
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
        out = isinstance(conv, torch.nn.modules.conv._ConvTransposeMixin)
    else:
        out = isinstance(conv, torch.nn.modules.conv._ConvTransposeNd)
    return out


def generate_noise(num_samples, z_sz, num_classes=0, rng_seed=None, device='cpu'):
    if isinstance(device, torch.nn.Module):
        device = next(device.parameters()).device
    rng, labels = None, None
    if rng_seed is None:
        z = torch.randn((num_samples, z_sz), device=device)
    else:
        rng = np.random.RandomState(rng_seed)
        z = rng.randn(num_samples, z_sz).astype(np.float32)
        z = torch.from_numpy(z).to(device, non_blocking=True)
    if num_classes > 0:
        if rng_seed is None:
            labels = torch.randint(low=0, high=num_classes, size=(num_samples,), device=device)
        else:
            labels = rng.randint(low=0, high=num_classes, size=(num_samples,), dtype=np.int)
            labels = torch.from_numpy(labels).to(device, non_blocking=True)
    return z, labels


def classification_accuracy(output, target, topk=(1,)):
    assert output.dim() == 2 and target.dim() == 1 and output.shape[0] == target.shape[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).int().sum(0, keepdim=True)
        res.append(correct_k)
    return res
