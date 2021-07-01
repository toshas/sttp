#!/usr/bin/env python
import json
import os
import random
import setuptools
import shutil
import time
import warnings

import numpy as np
import torch
import torch.utils.data
import wandb

from src.utils.config import parse_config_and_args, convert_to_namespace, format_dict
from src.utils.helpers import PersistentRandomSampler, net_extract_modules_order, get_singular_values_from_network, \
    tb_add_scalars, SilentSummaryWriter, get_statedict_num_params, \
    silent_torch_jit_trace_module, classification_accuracy, get_best_imgcls_metrics, verify_experiment_integrity
from src.utils.resolvers import resolve_optimizer, resolve_ops_factory, resolve_ops, resolve_lr_sched, \
    resolve_spectral_penalty, resolve_stiefel, resolve_imgcls_dataset, resolve_imgcls_model
from src.utils.spectral_tensors_factory import net_reparameterize_standard_to_factory, \
    get_canonical_singular_values_from_spectral_tensors_factory
from src.utils.visualizations import visualize_singular_values


def main(cfg):
    assert cfg.experiment == 'imgcls'

    seed = cfg.__dict__.get('random_seed', 2020)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    log_dir = cfg.log_dir
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

    dataset_train, dataset_valid, _ = resolve_imgcls_dataset(cfg)

    persistent_random_sampler = PersistentRandomSampler(
        dataset_train,
        cfg.num_training_steps * cfg.batch_size
    )

    ops_regular_dict = resolve_ops('regular')[0]
    ops_regular_classes = list(ops_regular_dict.values())

    model = resolve_imgcls_model(cfg.model_name)(cfg)

    dummy_input = (torch.randn(1, 3, 32, 32),)
    modules_order = net_extract_modules_order(
        model, dummy_input, ops_regular_classes, net_prefix=None
    )
    print('Modules order:' + '\n    '.join([''] + modules_order))

    factory = None
    if cfg.ops_use_factory:
        factory_cls = resolve_ops_factory(cfg.ops_factory_name)
        stiefel_full_cls = resolve_stiefel(cfg.ops_factory_stiefel_name, is_canonical=False)
        stiefel_canonical_cls = resolve_stiefel(
            cfg.ops_factory_stiefel_name, is_canonical=cfg.use_canonical_householder
        )
        factory = factory_cls(
            stiefel_full_cls,
            stiefel_canonical_cls,
            cfg.ops_factory_rank,
            **cfg.ops_factory_kwargs
        ).cuda()

    orig_num_weights = get_statedict_num_params(model.state_dict())
    compression_pct = 100
    if factory is not None:
        model, stats = net_reparameterize_standard_to_factory(
            model, factory, module_names_ignored=cfg.ops_factory_ignored, **cfg.__dict__.get('surgery_kwargs', {})
        )
        assert orig_num_weights == stats['nflt32_original_total']
        compression_pct = stats['compression_net_pct']
        print(f'Factory:\n{json.dumps(stats, indent=4)}\n{factory}')
    print(f'Model:\n{model}')

    compression_stats = {
        'orig_num_weights': orig_num_weights,
        'compression_pct': compression_pct,
    }

    if is_cuda:
        model = model.cuda()
        if factory is not None:
            factory = factory.cuda()

    if factory is not None:
        factory_jit = silent_torch_jit_trace_module(factory, {'forward': ()})

    def forward_factory():
        if factory is not None:
            weights = factory_jit.forward()
            factory.set_tensors(weights)

    param_groups = list(model.parameters())
    if factory is not None:
        param_groups += list(factory.parameters())
    optimizer = resolve_optimizer(cfg.optimizer)(param_groups, **cfg.optimizer_kwargs)

    lr_scheduler = resolve_lr_sched(optimizer, cfg.lr_sched, cfg.num_training_steps)

    step_loaded = 0
    metric_best = None
    metrics_best_all = None

    # load persistent state
    if os.path.isfile(checkpoint_path_latest):
        state_dict = torch.load(checkpoint_path_latest)
        model.load_state_dict(state_dict['model'])
        if factory is not None:
            factory.load_state_dict(state_dict['factory'])
        optimizer.load_state_dict(state_dict['optimizer'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        persistent_random_sampler.load_state_dict(state_dict['persistent_random_sampler'])
        step_loaded = state_dict['step']
        metric_best = state_dict['metric_best']
        metrics_best_all = state_dict['metrics_best_all']
        persistent_random_sampler.fast_forward_to(step_loaded * cfg.batch_size)
        if step_loaded == cfg.num_training_steps:
            print('Experiment was finished earlier; exiting')
            exit(0)

    # start dataloader workers
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=True,
        sampler=persistent_random_sampler,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=False,
    )

    # populate last_tensors
    with torch.no_grad():
        forward_factory()

    # training loop preamble
    if step_loaded == 0:
        print('Started training')
    else:
        print(f'Resumed training from step {step_loaded}')
    time_start_sec = time.monotonic()
    iter_train = iter(dataloader_train)

    # training loop
    with SilentSummaryWriter(tb_dir) as tb:
        tb_add_scalars(tb, 'compression', compression_stats)

        step = step_loaded
        while True:
            step += 1

            images, target = next(iter_train)
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # set models to training
            model.train()
            if factory is not None:
                factory.train()

            forward_factory()

            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, target)

            loss_spectral_penalty = None
            if factory is not None and cfg.spectral_penalty is not None:
                assert cfg.ops_use_factory, 'Spectral penalty is only available in factory mode'
                loss_spectral_penalty = resolve_spectral_penalty(cfg.spectral_penalty)(
                    factory.forward_singular_values()
                )
                loss = loss + cfg.spectral_penalty_weight * loss_spectral_penalty

            loss_stiefel_penalty = None
            if factory is not None and factory.has_stiefel_penalty:
                loss_stiefel_penalty = factory.stiefel_penalty()
                loss = loss + cfg.ops_factory_stiefel_penalty_weight * loss_stiefel_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            # update lr schedulers
            lr_scheduler.step()

            if step % cfg.num_log_loss_steps == 0:
                tb_add_scalars(tb, 'batch', {
                    'loss': loss,
                }, global_step=step)
                if loss_spectral_penalty is not None:
                    tb.add_scalar('batch/loss_spectral_penalty', loss_spectral_penalty, global_step=step)
                if loss_stiefel_penalty is not None:
                    tb.add_scalar('batch/loss_stiefel_penalty', loss_stiefel_penalty, global_step=step)
                tb_add_scalars(tb, 'progress', {
                    'lr': lr_scheduler.get_last_lr()[0],
                    'pct_done': 100 * step / cfg.num_training_steps,
                    'eta_hrs': (time.monotonic() - time_start_sec) * (cfg.num_training_steps - step) /
                               ((step - step_loaded) * 3600)
                }, global_step=step)

            if step % cfg.num_log_sv_steps == 0:
                # calculate/collect model singular values
                svs = get_singular_values_from_network(model, ops_regular_classes)
                if factory is not None and factory.have_singular_values:
                    svs.update(get_canonical_singular_values_from_spectral_tensors_factory(factory))

                # visualize and log svs
                if svs is not None and len(svs.keys()) > 0:
                    svs_unnorm_vis = visualize_singular_values(svs, modules_order,
                                                               cfg.vis_truncate_singular_values, False)
                    svs_norm_vis = visualize_singular_values(svs, modules_order,
                                                             cfg.vis_truncate_singular_values, True)
                    tb.add_image('singular_values/model', svs_unnorm_vis, global_step=step)
                    tb.add_image('singular_values_normalized/model', svs_norm_vis, global_step=step)
                    spectral_norm = {k: v[0].item() for k, v in svs.items()}
                    stable_rank = {k: ((v * v).sum() / (v[0] ** 2).clamp_min(1e-7)).item() for k, v in svs.items()}
                    tb.add_scalar('singular_values_stats/max_spectral_norm',
                                  max(spectral_norm.values()), global_step=step)
                    tb.add_scalar('singular_values_stats/max_stable_rank',
                                  max(stable_rank.values()), global_step=step)
                    tb_add_scalars(tb, 'spectral_norm', spectral_norm, global_step=step)
                    tb_add_scalars(tb, 'stable_rank', stable_rank, global_step=step)

            if step % cfg.num_checkpoint_steps == 0:
                model.eval()

                top1acc, top5acc = 0, 0
                for batch in dataloader_valid:
                    images, target = batch
                    images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    output = model(images)
                    acc1, acc5 = classification_accuracy(output, target, topk=(1, 5))
                    top1acc += acc1
                    top5acc += acc5

                metrics = {
                    'top1acc': 100. * top1acc.item() / len(dataset_valid),
                    'top5acc': 100. * top5acc.item() / len(dataset_valid),
                }

                metrics_best_all = get_best_imgcls_metrics(metrics, metrics_best_all)
                print(json.dumps(metrics, indent=4))
                tb_add_scalars(tb, 'metrics', metrics, global_step=step)
                tb_add_scalars(tb, 'metrics_best', metrics_best_all, global_step=step)

                metric_new_best = metrics_best_all['top1acc']
                have_new_best = metric_best != metric_new_best
                if have_new_best:
                    print(f'Step {step}: metric improved from {metric_best} to {metric_new_best}')
                    metric_best = metric_new_best
                else:
                    print(f'Step {step}: Metric did not improve')

                # prepare checkpoint
                state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'persistent_random_sampler': persistent_random_sampler.state_dict(),
                    'step': step,
                    'metric_best': metric_best,
                    'metrics_best_all': metrics_best_all,
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state_dict.update({
                        'lr_scheduler': lr_scheduler.state_dict(),
                    })
                if factory is not None:
                    state_dict['factory'] = factory.state_dict()
                torch.save(state_dict, checkpoint_path_latest + '.tmp')

                # handle best model artifacts
                if have_new_best:
                    # copy current checkpoint to best model checkpoint
                    shutil.copy(checkpoint_path_latest + '.tmp', checkpoint_path_best + '.tmp')
                    # commit best artifacts
                    os.rename(checkpoint_path_best + '.tmp', checkpoint_path_best)

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
