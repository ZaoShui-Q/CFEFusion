#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import time
import logging


def setup_logger(logpth):
    os.makedirs(logpth, exist_ok=True)

    logfile = 'Fusion-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)

    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        filename=logfile
    )
    logging.root.addHandler(logging.StreamHandler())