#!/usr/bin/python
# -*- coding:utf-8 -*-
import logging
import logging.config
import os
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

def get_logger(name='root'):
    conf_log = "cfg/logger_config.ini"
    logging.config.fileConfig(conf_log)
    return logging.getLogger(name)

ta_log = get_logger(__name__)
