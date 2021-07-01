#!/usr/bin/env python
import json
import os
import random
import setuptools
import shutil
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch.utils.data
import wandb
from torch_fidelity import calculate_metrics, GenerativeModelModuleWrapper
from torch_fidelity.utils import batch_interp
from torchvision.utils import make_grid

from src.modules.sngan import ConditionalBatchNorm2d
from src.utils.config import parse_config_and_args, convert_to_namespace, format_dict
from src.utils.helpers import PersistentRandomSampler, \
    net_extract_modules_order, get_singular_values_from_network, get_best_gan_metrics, \
    tb_add_scalars, SilentSummaryWriter, get_statedict_num_params, \
    silent_torch_jit_trace_module, ModuleEMA, generate_noise, verify_experiment_integrity
from src.utils.resolvers import resolve_optimizer, resolve_gan_models, resolve_ops_factory, \
    resolve_ops, resolve_lr_sched, resolve_gan_losses, resolve_gan_dataset, resolve_spectral_penalty, resolve_stiefel
from src.utils.spectral_compensation import spectral_compensation_stateful
from src.utils.spectral_tensors_factory import net_reparameterize_standard_to_factory, \
    get_canonical_singular_values_from_spectral_tensors_factory, net_reparameterize_factory_to_standard
from src.utils.visualizations import visualize_singular_values


def main(cfg):
    assert cfg.experiment == 'gan'

    seed = cfg.__dict__.get('random_seed', 2020)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    log_dir = cfg.log_dir
    generator_path_best = os.path.join(log_dir, 'generator_best.onnx')
    tb_dir = os.path.join(log_dir, 'tb')
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(cfg.wandb_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path_latest = os.path.join(checkpoints_dir, 'checkpoint_latest.pth')
    checkpoint_path_best = os.path.join(checkpoints_dir, 'checkpoint_best.pth')
    is_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None

    wandb.init(
        project=cfg.wandb_project,
        resume=True,
        name=cfg.experiment_name,
        id=cfg.experiment_name,
        config=cfg.__dict__,
        dir=cfg.wandb_dir,
        save_code=False,
    )
    wandb.tensorboard.patch(
        save=False,  # copies tb files into cloud and allows to run tensorboard in the cloud
        tensorboardX=False,
        pytorch=True,
    )

    # check the experiment is not resumed with different code and or settings
    verify_experiment_integrity(cfg)

    dataset, num_classes = resolve_gan_dataset(
        cfg.dataset, cfg.root_datasets[cfg.dataset], cfg.dataset_download,
        with_labels=cfg.conditioning, evaluation_transforms=False
    )
    ds_eval, _ = resolve_gan_dataset(
        cfg.dataset, cfg.root_datasets[cfg.dataset], cfg.dataset_download,
        with_labels=False, evaluation_transforms=True
    )
    if cfg.conditioning:
        assert type(num_classes) is int and num_classes > 0
    else:
        num_classes = 0

    persistent_random_sampler = PersistentRandomSampler(
        dataset,
        cfg.num_training_steps * cfg.batch_size * cfg.d_step_repeats
    )

    ops_regular_dict = resolve_ops('regular')[0]
    ops_regular_classes = list(ops_regular_dict.values())

    g_cls, d_cls = resolve_gan_models(cfg.model_name)
    g_loss, d_loss = resolve_gan_losses(cfg.loss_name)

    g_ops, g_fn_conv_back = resolve_ops(cfg.g_ops_name)
    g_ops_classes = list(g_ops.values())
    g_model = g_cls(cfg.model_preset, g_ops, num_classes=num_classes, **cfg.g_kwargs)
    z_sz = g_model.z_sz
    g_model_ema = None
    if cfg.g_ema:
        g_model_ema = ModuleEMA(g_model)
    with torch.no_grad():
        g_model_dummy_input = generate_noise(1, z_sz, num_classes)
        g_model_dummy_output = g_model.forward(*g_model_dummy_input)
    g_modules_order = net_extract_modules_order(
        g_model, g_model_dummy_input, ops_regular_classes + g_ops_classes, net_prefix=None,
        classes_ignored=(ConditionalBatchNorm2d,)
    )
    print('Generator modules order:' + '\n    '.join([''] + g_modules_order))
    g_factory, g_factory_ema = None, None
    if cfg.g_ops_use_factory:
        g_factory_cls = resolve_ops_factory(cfg.g_ops_factory_name)
        g_stiefel_full_cls = resolve_stiefel(cfg.g_ops_factory_stiefel_name, is_canonical=False)
        g_stiefel_canonical_cls = resolve_stiefel(
            cfg.g_ops_factory_stiefel_name, is_canonical=cfg.use_canonical_householder
        )
        g_factory = g_factory_cls(
            g_stiefel_full_cls,
            g_stiefel_canonical_cls,
            cfg.g_ops_factory_rank,
            **cfg.g_ops_factory_kwargs
        ).cuda()
        if cfg.g_ema:
            g_factory_ema = ModuleEMA(g_factory)

    d_ops, d_fn_conv_back = resolve_ops(cfg.d_ops_name)
    d_ops_classes = list(d_ops.values())
    d_model = d_cls(
        cfg.model_preset, d_ops,
        num_classes=num_classes, **cfg.d_kwargs
    )
    d_modules_order = net_extract_modules_order(
        d_model, (g_model_dummy_output, g_model_dummy_input[1]),
        ops_regular_classes + d_ops_classes, net_prefix=None,
        classes_ignored=(ConditionalBatchNorm2d,)
    )
    print('Discriminator modules order:' + '\n    '.join([''] + d_modules_order))
    d_factory = None
    if cfg.d_ops_use_factory:
        d_factory_cls = resolve_ops_factory(cfg.d_ops_factory_name)
        d_stiefel_full_cls = resolve_stiefel(cfg.d_ops_factory_stiefel_name, is_canonical=False)
        d_stiefel_canonical_cls = resolve_stiefel(
            cfg.d_ops_factory_stiefel_name, is_canonical=cfg.use_canonical_householder
        )
        d_factory = d_factory_cls(
            d_stiefel_full_cls,
            d_stiefel_canonical_cls,
            cfg.d_ops_factory_rank,
            **cfg.d_ops_factory_kwargs
        ).cuda()

    g_orig_num_weights = get_statedict_num_params(g_model.state_dict())
    g_compression_pct = 100
    if g_factory is not None:
        g_model, g_stats = net_reparameterize_standard_to_factory(
            g_model, g_factory, module_names_ignored=cfg.g_ops_factory_ignored, net_prefix=None,
            classes_ignored=(ConditionalBatchNorm2d,)
        )
        if cfg.g_ema:
            g_model_ema.dst, _ = net_reparameterize_standard_to_factory(
                g_model_ema.dst, g_factory_ema.dst, module_names_ignored=cfg.g_ops_factory_ignored, net_prefix=None,
                classes_ignored=(ConditionalBatchNorm2d,)
            )
        assert g_orig_num_weights == g_stats['nflt32_original_total']
        g_compression_pct = g_stats['compression_net_pct']
        g_fn_conv_back = net_reparameterize_factory_to_standard
        print(f'Generator factory:\n{json.dumps(g_stats, indent=4)}\n{g_factory}')
    print(f'Generator:\n{g_model}')

    d_orig_num_weights = get_statedict_num_params(d_model.state_dict())
    d_compression_pct = 100
    if d_factory is not None:
        d_model, d_stats = net_reparameterize_standard_to_factory(
            d_model, d_factory, module_names_ignored=cfg.d_ops_factory_ignored, net_prefix=None,
            classes_ignored=(ConditionalBatchNorm2d,)
        )
        assert d_orig_num_weights == d_stats['nflt32_original_total']
        d_compression_pct = d_stats['compression_net_pct']
        print(f'Discriminator:\n{json.dumps(d_stats, indent=4)}\n{d_factory}')
    print(f'Discriminator:\n{d_model}')

    compression_stats = {
        'generator_orig_num_weights': g_orig_num_weights,
        'generator_compression_pct': g_compression_pct,
        'discriminator_orig_num_weights': d_orig_num_weights,
        'discriminator_compression_pct': d_compression_pct,
    }

    if is_cuda:
        g_model = g_model.cuda()
        if cfg.g_ema:
            g_model_ema = g_model_ema.cuda()
        d_model = d_model.cuda()
        if g_factory is not None:
            g_factory = g_factory.cuda()
            if cfg.g_ema:
                g_factory_ema = g_factory_ema.cuda()
        if d_factory is not None:
            d_factory = d_factory.cuda()

    if g_factory is not None:
        g_factory_jit = silent_torch_jit_trace_module(g_factory, {'forward': ()})
    if d_factory is not None:
        d_factory_jit = silent_torch_jit_trace_module(d_factory, {'forward': ()})

    def forward_g_factory():
        if g_factory is not None:
            g_weights = g_factory_jit.forward()
            g_factory.set_tensors(g_weights)

    def forward_d_factory():
        if d_factory is not None:
            d_weights = d_factory_jit.forward()
            d_factory.set_tensors(d_weights)

    def forward_g_factory_ema():
        if g_factory_ema is not None:
            g_weights_ema = g_factory_ema.forward()
            g_factory_ema.dst.set_tensors(g_weights_ema)

    g_param_groups = list(g_model.parameters())
    if g_factory is not None:
        g_param_groups += list(g_factory.parameters())
    g_optimizer = resolve_optimizer(cfg.g_optimizer)(g_param_groups, **cfg.g_optimizer_kwargs)

    d_param_groups = list(d_model.parameters())
    if d_factory is not None:
        d_param_groups += list(d_factory.parameters())
    d_optimizer = resolve_optimizer(cfg.d_optimizer)(d_param_groups, **cfg.d_optimizer_kwargs)

    g_lr_scheduler = resolve_lr_sched(g_optimizer, cfg.lr_sched, cfg.num_training_steps)
    d_lr_scheduler = resolve_lr_sched(d_optimizer, cfg.lr_sched, cfg.num_training_steps)

    g_spectral_compensation_state = None
    if cfg.g_spectral_compensation_frequency > 0:
        assert cfg.g_ops_name == 'regular' and not cfg.g_ops_use_factory, 'Incorrect combination of settings'
        g_spectral_compensation_state = {}

    d_spectral_compensation_state = None
    if cfg.d_spectral_compensation_frequency > 0:
        assert cfg.d_ops_name == 'regular' and not cfg.d_ops_use_factory, 'Incorrect combination of settings'
        d_spectral_compensation_state = {}

    step_loaded = 0
    metric_best = None
    metrics_best_all = None

    # load persistent state
    if os.path.isfile(checkpoint_path_latest):
        state_dict = torch.load(checkpoint_path_latest)
        g_model.load_state_dict(state_dict['g_model'])
        d_model.load_state_dict(state_dict['d_model'])
        if cfg.g_ema:
            g_model_ema.load_state_dict(state_dict['g_model_ema'])
        if g_factory is not None:
            g_factory.load_state_dict(state_dict['g_factory'])
            if cfg.g_ema:
                g_factory_ema.load_state_dict(state_dict['g_factory_ema'])
        if d_factory is not None:
            d_factory.load_state_dict(state_dict['d_factory'])
        g_optimizer.load_state_dict(state_dict['g_optimizer'])
        d_optimizer.load_state_dict(state_dict['d_optimizer'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_lr_scheduler.load_state_dict(state_dict['g_lr_scheduler'])
            d_lr_scheduler.load_state_dict(state_dict['d_lr_scheduler'])
        if g_spectral_compensation_state is not None:
            g_spectral_compensation_state = state_dict['g_spectral_compensation_state']
        if d_spectral_compensation_state is not None:
            d_spectral_compensation_state = state_dict['d_spectral_compensation_state']
        persistent_random_sampler.load_state_dict(state_dict['persistent_random_sampler'])
        step_loaded = state_dict['step']
        metric_best = state_dict['metric_best']
        metrics_best_all = state_dict['metrics_best_all']
        persistent_random_sampler.fast_forward_to(step_loaded * cfg.batch_size * cfg.d_step_repeats)
        if step_loaded == cfg.num_training_steps:
            print('Experiment was finished earlier; exiting')
            exit(0)

    # start dataloader workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=True,
        sampler=persistent_random_sampler,
    )

    # populate last_tensors
    with torch.no_grad():
        forward_g_factory()
        forward_d_factory()

    # training loop preamble
    if step_loaded == 0:
        print('Started training')
    else:
        print(f'Resumed training from step {step_loaded}')
    time_start_sec = time.monotonic()
    dataiter = iter(dataloader)

    # training loop
    with SilentSummaryWriter(tb_dir) as tb:
        tb_add_scalars(tb, 'compression', compression_stats)

        step = step_loaded
        while True:
            step += 1

            # set models to training
            g_model.train()
            d_model.train()
            if g_factory is not None:
                g_factory.train()
            if d_factory is not None:
                d_factory.train()

            loss_d, loss_g = None, None
            loss_g_spectral_penalty, loss_d_spectral_penalty = None, None
            loss_g_stiefel_penalty, loss_d_stiefel_penalty = None, None

            #
            # update generator
            #

            g_model.requires_grad_(True)
            d_model.requires_grad_(False)
            if d_factory is not None:
                d_factory.last_tensors = tuple(a.detach() for a in d_factory.last_tensors)

            forward_g_factory()
            fake_z = generate_noise(cfg.batch_size, z_sz, num_classes, device=g_model)
            fake_rgb = g_model.forward(*fake_z)
            fake_out = d_model.forward(fake_rgb, fake_z[1])
            loss_g = g_loss(fake_out, fake_z[1])

            if cfg.g_spectral_penalty is not None:
                assert cfg.g_ops_use_factory, 'Spectral penalty is only available in factory mode'
                loss_g_spectral_penalty = resolve_spectral_penalty(cfg.g_spectral_penalty)(
                    g_factory.forward_singular_values()
                )
                loss_g = loss_g + cfg.g_spectral_penalty_weight * loss_g_spectral_penalty

            if g_factory is not None and g_factory.has_stiefel_penalty:
                loss_g_stiefel_penalty = g_factory.stiefel_penalty()
                loss_g = loss_g + cfg.g_ops_factory_stiefel_penalty_weight * loss_g_stiefel_penalty

            g_optimizer.zero_grad()
            loss_g.backward()
            g_optimizer.step()
            loss_g = loss_g.item()

            if g_spectral_compensation_state is not None and step % cfg.g_spectral_compensation_frequency == 0:
                g_spectral_compensation_state = spectral_compensation_stateful(
                    g_model,
                    state=g_spectral_compensation_state,
                    classes=(ops_regular_dict['cls_conv2d'], ops_regular_dict['cls_linear'],),
                    **cfg.g_spectral_compensation_kwargs
                )

            #
            # update discriminator
            #
            for d_step in range(cfg.d_step_repeats):
                real = next(dataiter)
                if is_cuda:
                    if type(real) in (tuple, list):
                        real = tuple(a.cuda(non_blocking=True) for a in real)
                    else:
                        real = real.cuda(non_blocking=True)
                if not cfg.conditioning:
                    real = real, None

                g_model.requires_grad_(False)
                if g_factory is not None:
                    g_factory.last_tensors = tuple(a.detach() for a in g_factory.last_tensors)
                d_model.requires_grad_(True)

                forward_d_factory()
                real_out = d_model.forward(*real)
                fake_z = generate_noise(cfg.batch_size, z_sz, num_classes, device=g_model)
                fake_rgb = g_model.forward(*fake_z)
                fake_out = d_model.forward(fake_rgb, fake_z[1])
                loss_d = d_loss(fake_out, real_out, real[1])

                if cfg.d_spectral_penalty is not None:
                    assert cfg.d_ops_use_factory, 'Spectral penalty is only available in factory mode'
                    loss_d_spectral_penalty = resolve_spectral_penalty(cfg.d_spectral_penalty)(
                        d_factory.forward_singular_values()
                    )
                    loss_d = loss_d + cfg.d_spectral_penalty_weight * loss_d_spectral_penalty

                if d_factory is not None and d_factory.has_stiefel_penalty:
                    loss_d_stiefel_penalty = d_factory.stiefel_penalty()
                    loss_d = loss_d + cfg.d_ops_factory_stiefel_penalty_weight * loss_d_stiefel_penalty

                d_optimizer.zero_grad()
                loss_d.backward()
                d_optimizer.step()
                loss_d = loss_d.item()

                if d_spectral_compensation_state is not None and \
                        (step * cfg.d_step_repeats + d_step) % cfg.d_spectral_compensation_frequency == 0:
                    d_spectral_compensation_state = spectral_compensation_stateful(
                        d_model,
                        state=d_spectral_compensation_state,
                        classes=(ops_regular_dict['cls_conv2d'], ops_regular_dict['cls_linear'],),
                        **cfg.d_spectral_compensation_kwargs
                    )

            if cfg.g_ema:
                g_model_ema.update()
                if g_factory is not None:
                    g_factory_ema.update()

            # update lr schedulers
            g_lr_scheduler.step()
            d_lr_scheduler.step()

            if step % cfg.num_log_loss_steps == 0:
                tb_add_scalars(tb, 'batch', {
                    'loss_g': loss_g,
                    'loss_d': loss_d,
                }, global_step=step)
                if loss_g_spectral_penalty is not None:
                    tb.add_scalar('batch/loss_g_spectral_penalty', loss_g_spectral_penalty, global_step=step)
                if loss_d_spectral_penalty is not None:
                    tb.add_scalar('batch/loss_d_spectral_penalty', loss_d_spectral_penalty, global_step=step)
                if loss_g_stiefel_penalty is not None:
                    tb.add_scalar('batch/loss_g_stiefel_penalty', loss_g_stiefel_penalty, global_step=step)
                if loss_d_stiefel_penalty is not None:
                    tb.add_scalar('batch/loss_d_stiefel_penalty', loss_d_stiefel_penalty, global_step=step)
                tb_add_scalars(tb, 'progress', {
                    'lr_g': g_lr_scheduler.get_last_lr()[0],
                    'lr_d': d_lr_scheduler.get_last_lr()[0],
                    'pct_done': 100 * step / cfg.num_training_steps,
                    'eta_hrs': (time.monotonic() - time_start_sec) * (cfg.num_training_steps - step) /
                               ((step - step_loaded) * 3600)
                }, global_step=step)

            if step % cfg.num_log_images_steps == 0:
                GH, GW = 8, 8
                if cfg.conditioning:
                    GH = min(num_classes, 10)
                GT = GH * GW
                plot_grid = partial(make_grid, pad_value=0, nrow=GW)
                forward_g_factory_ema()
                g = g_model_ema.dst if cfg.g_ema else g_model
                device = next(g.parameters()).device
                g.eval()
                with torch.no_grad():
                    for name, seed in (('fixed', 2020), ('random', None)):
                        # random samples, grouped by class if we are in conditional mode
                        z, l = generate_noise(GT, z_sz, num_classes, rng_seed=seed, device=g)
                        if cfg.conditioning:
                            if GH != num_classes:
                                l = l[:GH].view(GH, 1).repeat(1, GW).view(GT, )
                            else:
                                l = torch.arange(num_classes, device=device).view(GH, 1).repeat(1, GW).view(GT, )
                        rgb = g.forward(z, l)
                        padding = rgb.shape[-1] // 16
                        tb.add_image(f'samples_{name}', plot_grid(rgb, padding=padding), global_step=step)

                        # interpolation sheets
                        z = z.view(GH, GW, z_sz)
                        z0, z1 = z[:, 0, :], z[:, -1, :]
                        z = [batch_interp(z0, z1, t / (GW-1), cfg.z_interp_mode).unsqueeze(1) for t in range(GW)]
                        z = torch.cat(z, dim=1).reshape(GT, z_sz)
                        rgb = g.forward(z, l)
                        tb.add_image(f'interpolation_{name}', plot_grid(rgb, padding=padding), global_step=step)

            if step % cfg.num_log_sv_steps == 0:
                # calculate generator singular values
                g_svs = get_singular_values_from_network(
                    g_model, g_ops_classes + ops_regular_classes, classes_ignored=(ConditionalBatchNorm2d,)
                )
                if g_factory is not None and g_factory.have_singular_values:
                    g_svs.update(get_canonical_singular_values_from_spectral_tensors_factory(g_factory))

                # calculate discriminator singular values
                d_svs = get_singular_values_from_network(
                    d_model, d_ops_classes + ops_regular_classes, classes_ignored=(ConditionalBatchNorm2d,)
                )
                if d_factory is not None and d_factory.have_singular_values:
                    d_svs.update(get_canonical_singular_values_from_spectral_tensors_factory(d_factory))

                # visualize and log svs
                if g_svs is not None and len(g_svs.keys()) > 0:
                    g_svs_unnorm_vis = visualize_singular_values(g_svs, g_modules_order,
                                                                 cfg.vis_truncate_singular_values, False)
                    g_svs_norm_vis = visualize_singular_values(g_svs, g_modules_order,
                                                               cfg.vis_truncate_singular_values, True)
                    tb.add_image('singular_values/generator', g_svs_unnorm_vis, global_step=step)
                    tb.add_image('singular_values_normalized/generator', g_svs_norm_vis, global_step=step)
                    g_spectral_norm = {k: v[0].item() for k, v in g_svs.items()}
                    g_stable_rank = {k: ((v * v).sum() / (v[0] ** 2).clamp_min(1e-7)).item() for k, v in g_svs.items()}
                    tb.add_scalar('singular_values_stats/generator_max_spectral_norm',
                                  max(g_spectral_norm.values()), global_step=step)
                    tb.add_scalar('singular_values_stats/generator_max_stable_rank',
                                  max(g_stable_rank.values()), global_step=step)
                    tb_add_scalars(tb, 'generator_spectral_norm', g_spectral_norm, global_step=step)
                    tb_add_scalars(tb, 'generator_stable_rank', g_stable_rank, global_step=step)

                if d_svs is not None and len(d_svs.keys()) > 0:
                    d_svs_unnorm_vis = visualize_singular_values(d_svs, d_modules_order,
                                                                 cfg.vis_truncate_singular_values, False)
                    d_svs_norm_vis = visualize_singular_values(d_svs, d_modules_order,
                                                               cfg.vis_truncate_singular_values, True)
                    tb.add_image('singular_values/discriminator', d_svs_unnorm_vis, global_step=step)
                    tb.add_image('singular_values_normalized/discriminator', d_svs_norm_vis, global_step=step)
                    d_spectral_norm = {k: v[0].item() for k, v in d_svs.items()}
                    d_stable_rank = {k: ((v * v).sum() / (v[0] ** 2).clamp_min(1e-7)).item() for k, v in d_svs.items()}
                    tb.add_scalar('singular_values_stats/discriminator_max_spectral_norm',
                                  max(d_spectral_norm.values()), global_step=step)
                    tb.add_scalar('singular_values_stats/discriminator_max_stable_rank',
                                  max(d_stable_rank.values()), global_step=step)
                    tb_add_scalars(tb, 'discriminator_spectral_norm', d_spectral_norm, global_step=step)
                    tb_add_scalars(tb, 'discriminator_stable_rank', d_stable_rank, global_step=step)

            if step % cfg.num_checkpoint_steps == 0:
                g_model.eval()
                d_model.eval()
                forward_g_factory_ema()

                # calculate metrics
                metrics = calculate_metrics(
                    input1=GenerativeModelModuleWrapper(
                        g_model_ema.dst if cfg.g_ema else g_model,
                        z_sz,
                        cfg.z_type,
                        num_classes
                    ),
                    input1_model_num_samples=cfg.num_validation_images,
                    input2=ds_eval,
                    cache_input2_name=f'specific_{cfg.dataset}_'
                                      f'{"conditional" if cfg.conditioning else "unconditional"}',
                    cuda=is_cuda,
                    save_cpu_ram=True,
                    ppl_z_interp_mode=cfg.z_interp_mode,
                    **cfg.fidelity_kwargs
                )
                metrics_best_all = get_best_gan_metrics(metrics, metrics_best_all)
                print(json.dumps(metrics, indent=4))
                tb_add_scalars(tb, 'metrics', metrics, global_step=step)
                tb_add_scalars(tb, 'metrics_best', metrics_best_all, global_step=step)

                metric_new_best = metrics_best_all[cfg.fidelity_leading_metric_name]
                have_new_best = metric_best != metric_new_best
                if have_new_best:
                    print(f'Step {step}: metric improved from {metric_best} to {metric_new_best}')
                    metric_best = metric_new_best
                else:
                    print(f'Step {step}: Metric did not improve')

                # prepare checkpoint
                state_dict = {
                    'g_model': g_model.state_dict(),
                    'd_model': d_model.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'persistent_random_sampler': persistent_random_sampler.state_dict(),
                    'step': step,
                    'metric_best': metric_best,
                    'metrics_best_all': metrics_best_all,
                }
                if cfg.g_ema:
                    state_dict['g_model_ema'] = g_model_ema.state_dict()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state_dict.update({
                        'g_lr_scheduler': g_lr_scheduler.state_dict(),
                        'd_lr_scheduler': d_lr_scheduler.state_dict(),
                    })
                if g_spectral_compensation_state is not None:
                    state_dict['g_spectral_compensation_state'] = g_spectral_compensation_state
                if d_spectral_compensation_state is not None:
                    state_dict['d_spectral_compensation_state'] = d_spectral_compensation_state
                if g_factory is not None:
                    state_dict['g_factory'] = g_factory.state_dict()
                    if cfg.g_ema:
                        state_dict['g_factory_ema'] = g_factory_ema.state_dict()
                if d_factory is not None:
                    state_dict['d_factory'] = d_factory.state_dict()
                torch.save(state_dict, checkpoint_path_latest + '.tmp')

                # handle best model artifacts
                if have_new_best:
                    # copy current checkpoint to best model checkpoint
                    shutil.copy(checkpoint_path_latest + '.tmp', checkpoint_path_best + '.tmp')

                    # convert generator model to use regular convolutions and dump it to onnx
                    g = g_model_ema.dst if cfg.g_ema else g_model
                    if g_fn_conv_back is not None:
                        factory = None
                        if g_factory is not None:
                            factory = g_factory_ema.dst if cfg.g_ema else g_factory
                        g = g_fn_conv_back(
                            g,
                            inplace=False,
                            net_prefix=None,
                            factory=factory,
                        )

                    dynamic_axes = {'z': {0: 'batch'}, 'rgb': {0: 'batch'}}
                    g_model_dummy_input = generate_noise(1, z_sz, num_classes, device=g)
                    if cfg.conditioning:
                        input_names = ['z', 'condition']
                        dynamic_axes['condition'] = {0: 'batch'}
                    else:
                        input_names = ['z']
                        g_model_dummy_input = g_model_dummy_input[0]
                    torch.onnx.export(
                        g, g_model_dummy_input,
                        generator_path_best + '.tmp',
                        opset_version=11,  # minimum for proper upsampling
                        input_names=input_names, output_names=['rgb'],
                        dynamic_axes=dynamic_axes,
                    )

                    # commit best artifacts
                    os.rename(checkpoint_path_best + '.tmp', checkpoint_path_best)
                    os.rename(generator_path_best + '.tmp', generator_path_best)

                # commit checkpoint
                os.rename(checkpoint_path_latest + '.tmp', checkpoint_path_latest)

            if step == cfg.num_training_steps:
                break

    print(f'Step {step}: finished training')


if __name__ == '__main__':
    cfg, cfg_warnings = parse_config_and_args()
    cfg = convert_to_namespace(cfg)
    if len(cfg_warnings) > 0:
        print('\n'.join(cfg_warnings))
    print(format_dict(cfg.__dict__))
    main(cfg)
