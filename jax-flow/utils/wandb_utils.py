"""WandB setup helper."""

import wandb
import tempfile
import datetime
import time
import os
import numpy as np
import absl.flags as flags
import ml_collections
from ml_collections.config_dict import FieldReference


def get_flag_dict():
    d = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
    for k in d:
        if isinstance(d[k], ml_collections.ConfigDict):
            d[k] = d[k].to_dict()
    return d


def default_wandb_config():
    cfg = ml_collections.ConfigDict()
    cfg.offline = False
    cfg.project = "fsdit"
    cfg.entity = FieldReference(None, field_type=str)
    group = FieldReference(None, field_type=str)
    cfg.exp_prefix = group
    cfg.group = group
    name = FieldReference(None, field_type=str)
    cfg.name = name
    cfg.exp_descriptor = name
    cfg.unique_identifier = ""
    cfg.random_delay = 0
    return cfg


def setup_wandb(hyperparam_dict, entity=None, project="fsdit", group=None, name=None,
                unique_identifier="", offline=False, random_delay=0, **kw):
    kw.pop("exp_descriptor", None)
    kw.pop("exp_prefix", None)

    if not unique_identifier:
        if random_delay:
            time.sleep(np.random.uniform(0, random_delay))
        flag_dict = get_flag_dict()
        unique_identifier = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{np.random.randint(0, 1e6):06d}"
        )
        if 'seed' in flag_dict:
            unique_identifier += f"_{flag_dict['seed']:02d}"

    if name:
        name = name.format(**{**get_flag_dict(), **hyperparam_dict}).replace("/", "_")

    exp_id = f"{name}_{unique_identifier}" if name else None
    out_dir = "/nfs/wandb" if os.path.exists("/nfs/wandb") else tempfile.mkdtemp()
    tags = [group] if group else None

    run = wandb.init(
        config=hyperparam_dict, project=project, entity=entity,
        tags=tags, group=group, dir=out_dir, id=exp_id, name=name,
        settings=wandb.Settings(start_method="thread", _disable_stats=False),
        mode="offline" if offline else "online", save_code=True, **kw,
    )
    flag_dict = {k: v for k, v in get_flag_dict().items() if v is not None}
    wandb.config.update(flag_dict, allow_val_change=True)
    return run
