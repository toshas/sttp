#!/usr/bin/env python
from src import train_gan, train_imgcls
from src.utils.config import parse_config_and_args, convert_to_namespace, format_dict


if __name__ == '__main__':
    cfg, cfg_warnings = parse_config_and_args()
    cfg = convert_to_namespace(cfg)
    if len(cfg_warnings) > 0:
        print('\n'.join(cfg_warnings))
    print(format_dict(cfg.__dict__))
    {
        'gan': train_gan.main,
        'imgcls': train_imgcls.main
    }[cfg.experiment](cfg)
