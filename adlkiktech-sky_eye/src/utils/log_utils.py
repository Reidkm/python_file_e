#!/usr/bin/python
# -*- coding:utf-8 -*-
import time, re
from globalLog import ta_log

def printLog(content, isBacklog=False) :
    print(content)
    ta_log.info(content)
