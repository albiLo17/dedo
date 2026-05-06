"""
Common utilities for training.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
from datetime import datetime
import numpy as np

import os
import platform
import torch
import wandb


def object_to_str(obj):
    # Print all fields of the given object as text in tensorboard.
    text_str = ''
    for member in vars(obj):
        # Tensorboard uses markdown-like formatting, hence '  \n'.
        text_str += '  \n{:s}={:s}'.format(
            str(member), str(getattr(obj, member)))
    return text_str


def init_train(algo, args, tags=None, wandb_resume_id=None):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    if platform.system() == 'Linux':
        os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'
    logdir = None
    if args.logdir is not None:
        tstamp = datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
        lst = [algo, tstamp, args.env]
        subdir = '_'.join(lst)
        logdir = os.path.join(os.path.expanduser(args.logdir), subdir)
        if args.use_wandb:
            wandb_kwargs = dict(config=vars(args), project='dedo',
                                name=logdir, tags=tags,
                                sync_tensorboard=True)
            if wandb_resume_id:
                # Continue the same wandb run instead of starting a fresh
                # one. wandb keeps the original run name; `name=logdir` is
                # ignored. resume="must" raises if the ID doesn't exist,
                # which is the safer behavior — silent fall-through to a
                # new run would defeat the purpose.
                wandb_kwargs['id'] = wandb_resume_id
                wandb_kwargs['resume'] = 'must'
            wandb.init(**wandb_kwargs)
            try:  # patch only once, if more than one run, ignore error
                wandb.tensorboard.patch()
            except Exception:
                pass
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    return logdir, device
