import argparse
import ast
import json
import os
import re
import sys

import yaml


def dict_deep_update(what, patch, prefix=None, overwrite_warns=None):
    for k in patch.keys():
        new_prefix = k if prefix is None else prefix + '/' + k
        if isinstance(patch[k], dict):
            if k in what:
                assert isinstance(what[k], dict)
            else:
                what[k] = dict()
            dict_deep_update(what[k], patch[k], prefix=new_prefix, overwrite_warns=overwrite_warns)
        else:
            if k in what and overwrite_warns is not None:
                overwrite_warns.append('Overwriting key \'{}\': old \'{}\' new \'{}\''
                      .format(new_prefix, str(what[k]), str(patch[k])))
            what[k] = patch[k]


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


def dict_deep_put(d, key_path, val, split_ch='/', dict_type=dict):
    if type(key_path) is str:
        parts = key_path.split(split_ch)
    elif type(key_path) is list:
        parts = key_path
    else:
        assert False
    for i, part in enumerate(parts):
        is_last = (i == len(parts)-1)
        if is_last:
            d[part] = val
        else:
            if part not in d:
                d[part] = dict_type()
            d = d[part]


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def parse_config_and_args(as_namespace=False):
    if '--help' in sys.argv or '-h' in sys.argv:
        app = os.path.basename(sys.argv[0])
        print(
            f'{app} --cfg <path.yml> [--cfg <path.yml>] [--node-path-key <value>] [--node-path <dict>]\n' +
            'Multiple configs and command line keys act as deep tree patches.\n' +
            'Warnings will be issued only for keys overriden by subsequent config files or CLI.\n' +
            'Lists are treated immutably, subsequent definition overrides entire list.\n' +
            'If runtime complains about missing attribute in config, just specify using the most convenient way.\n' +
            'Examples: \n' +
            '  train.py --cfg ~/semseg.yml\n' +
            '  train.py --cfg ~/env_dgx.yml --cfg ~/semseg.yml\n' +
            '  train.py --cfg ~/semseg.yml --optimizer_kwargs-momentum 0.9\n' +
            '  train.py --cfg ~/semseg.yml --optimizer_kwargs \'{"momentum":0.9,"dampening":0.1}\'\n'
        )
        exit(0)

    cfg = dict()

    # fix yaml scientific notation loader
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    argv = []
    for arg in sys.argv[1:]:
        assert arg.count('=') in (0, 1)
        if arg.count('=') == 1:
            pos_eq = arg.find('=')
            arg, val = arg[:pos_eq], arg[pos_eq+1:]
            assert len(val) > 0
            argv.append(arg)
            argv.append(val)
        else:
            argv.append(arg)

    cfg_warnings = []
    last_key = None
    for arg in argv:
        if len(arg) >= 2 and arg[0:2] == '--':
            assert last_key is None, 'Key --{} must have a value'.format(last_key)
            last_key = arg[2:]
            continue
        assert last_key is not None, '{} is ambiguous, must begin with \'--\' if key, key missing if value'
        val_str = arg
        if last_key in ['config', 'cfg']:
            cfg_path = expandpath(val_str)
            if not os.path.isfile(cfg_path):
                cfg_path_proposal = os.path.join(os.path.dirname(sys.argv[0]), cfg_path)
                if os.path.isfile(cfg_path_proposal):
                    cfg_path = cfg_path_proposal
                else:
                    assert False, 'Config file not found in \'{}\''.format(val_str)
            cfg_filenamebase = os.path.splitext(os.path.basename(cfg_path))[0]
            with open(cfg_path) as fp:
                data = fp.read().replace('__FILENAMEBASE__', cfg_filenamebase)
                cfg_patch = yaml.safe_load(data)
            assert type(cfg_patch) is dict, 'Config file \'{}\' must be a YAML dict'
            dict_deep_update(cfg, cfg_patch, overwrite_warns=cfg_warnings)
            last_key = None
            continue
        try:
            val_ast = ast.literal_eval(val_str)
            if type(val_ast) is dict:
                cfg_node = dict_deep_get(cfg, last_key, default=dict(), create_if_missing=True)
                dict_deep_update(cfg_node, val_ast)
            elif last_key.count('.') > 0 and last_key.count('-') == 0:
                dict_deep_put(cfg, last_key, val_ast, split_ch='.')
            else:
                dict_deep_put(cfg, last_key, val_ast, split_ch='-')
        except (SyntaxError, ValueError):
            if last_key.count('.') > 0 and last_key.count('-') == 0:
                dict_deep_put(cfg, last_key, val_str, split_ch='.')
            else:
                dict_deep_put(cfg, last_key, val_str, split_ch='-')
        last_key = None

    if cfg.get("assert_env_set", None) is not None:
        for e in cfg["assert_env_set"]:
            assert e in os.environ, 'Environment variable "{}" not set'.format(e)

    if 'log_dir' in cfg:
        if 'root_wandb' in cfg.keys():
            cfg['wandb_dir'] = expandpath(os.path.join(cfg['root_wandb'], cfg['log_dir']))
        if 'root_logs' in cfg.keys():
            cfg['log_dir'] = expandpath(os.path.join(cfg['root_logs'], cfg['log_dir']))

    # convert dict to a one-tier namespace
    if as_namespace:
        cfg = convert_to_namespace(cfg)

    return cfg, cfg_warnings


def convert_to_namespace(cfg):
    class GlobalConfig(argparse.Namespace):
        pass

    namespace = GlobalConfig()
    for k, v in cfg.items():
        setattr(namespace, k, v)
    return namespace


def format_dict(d):
    return json.dumps(d, indent=4)
