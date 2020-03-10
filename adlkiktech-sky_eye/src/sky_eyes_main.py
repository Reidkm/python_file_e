#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division

import threading
import time
from multiprocessing import Process
import asyncore
import socket
import os
import sys
import cv2
import numpy as np

from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels,depth_LUT,k_LUT,b_LUT,cal_area_center
from option import Options
import datetime
import Queue
import traceback
import binascii
import math
from message import Message
from matplotlib import pyplot as plt
import random
import yaml

from utils import log_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_height = 0
image_width = 0
max_width = 0
img_sent_count = 0
img_recv_count = 0
IS_DEBUG = False
TCP_FLAG = True
TCP_SEND = True

# host = '192.168.1.185'
host = '127.0.0.1'
port = 9002
tick = datetime.datetime.now()

STOP_FLAG = False
IS_RECORD = False


def printHex(content):
    return binascii.hexlify(bytearray(content))


def printTick(content):
    global tick
    tick2 = datetime.datetime.now()
    print ' ', content, ' ', tick2 - tick
    tick = tick2


class boss():
    def __init__(self):
        data = list_manage()
        lock = threading.Lock()
        self.data = data
        self.client = None
        if TCP_FLAG:
            self.client = HTTPClient(host, port, self)
        self.c0 = consume(data, 'c0', lock, self.client)
        self.p = product(data, 'p', lock)
        self.c0.start()
        self.p.start()
        self.go()

    def go(self):
        while not STOP_FLAG:
            time.sleep(1)

    def retry_connect(self):
        self.client = HTTPClient(host, port, self)
        self.c0.setClient(self.client)
        self.client.startConnect()


class HTTPClient(asyncore.dispatcher):
    def __init__(self, host, port, boss):
        asyncore.dispatcher.__init__(self)
        self.host = host
        self.port = port
        self.buffer = Queue.Queue()
        self.receive_buffer = Queue.Queue()
        self.boss = boss

    def startConnect(self):
        print('start connect in http client')
        try:
            self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connect((self.host, self.port))
        except Exception as exc:
            traceback.print_exc()
            print(exc)

    def initate_connection_with_server(self):
        print("trying to initialize connection with server...")
        asyncore.dispatcher.__init__(self)
        self.startConnect()

    def handle_connect(self):
        print('connected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # sys.exit("connected")

    def handle_error(self):
        print "error!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # self.initate_connection_with_server()

    def handle_close(self):
        self.close()
        print("connection close!!!!")
        if not STOP_FLAG:
            threading.Timer(1, self.boss.retry_connect).start()

    def handle_read(self):
        # print('received')
        received_data = self.recv(32)
        print(received_data)

    def writable(self):
        return (self.buffer.qsize() > 0)

    def handle_write(self):
        global img_sent_count
        try:
            content = self.buffer.get()
            if content is None:
                print 'nothing to sent'
            img_sent_count += 1
            print 'content.len', len(content)
            if TCP_SEND:
                self.send(content)
        except Exception as e:
            print e
            traceback.print_exc()
            pass

    def run_command(self, cmd):
        # print "TCP add to buffer"
        if self.buffer:
            self.buffer.put(cmd)


class consume(threading.Thread):
    def __init__(self, data, id, lock, client):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.lock = lock
        self.client = client

    def produceData(self, context):
        checksum = sum(context)
        context.extend((checksum % 256,))
        return context

    def setClient(self, client):
        self.client = client

    def run(self):
        while not STOP_FLAG:
            time.sleep(0.01)
            self.lock.acquire()
            context = self.data.get()

            if not (context is None):
                # print 'send via TCP'
                # a = np.ones((83886,1), dtype=np.uint8) * 127
                if not (self.client is None):
                    self.client.run_command(context)

            try:
                asyncore.poll(timeout=.001)
            except Exception as e:
                print e
                pass
            self.lock.release()


class product(threading.Thread):
    def __init__(self, data, id, lock):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.lock = lock

    def process(self, args, mirror=False):
        print('Runing............')

        # Define the codec and create VideoWriter object
        height = 480
        width = 640
        fps = FPS().start()
        stopflag = False
        idx = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        eagle_list = [{'id': 0, 'source': 'rtsp://admin:admin@127.0.0.1/out.h264',
            'camera_param':'web_camera_para_2018_12_30.yaml'}
                # {'id':1, 'source':'rtsp://admin:kiktech2016@192.168.1.65:554/out.h264'}
                ]
        eagles = []
        from top_eagle import TopEagle
        for eagleinfo in eagle_list:
            eagle = TopEagle(eagleinfo['id'], eagleinfo['source'],
                eagleinfo['camera_param'], fix_distortion=True, need_record=IS_RECORD)
            eagles.append(eagle)
            # cv2.namedWindow('Camera {0}'.format(eagleinfo['id']))

        for eagle in eagles:
            eagle.cameraSwitch()

        while not stopflag:
            idx += 1

            for eagle in eagles:
                showimg = np.zeros((height, width, 3))
                msg = None
                result = eagle.process(use_nn=True)
                if result is None:
                    continue

                if not (result is None):
                    msg = result[0]
                    showimg = result[1]['result']

                    if IS_RECORD and result[1]['src'] is not None:
                        eagle.record(result[1]['src'])

                    if showimg is None:
                        showimg = result[1]['src']

                else:
                    continue

                # send out
                if msg is not None:
                    self.lock.acquire()
                    self.data.add(msg.get_transportable())
                    self.lock.release()
                fps.update()

                if not IS_RECORD:
                    text2 = '[INFO] approx. FPS: {:.2f}'.format(fps.fps())
                    cv2.putText(
                        showimg, text2, (0, 20), font, 0.9, (255, 255, 255), 3)

                cv2.imshow('Camera {0}'.format(eagle.id), showimg)

                key = cv2.waitKey(1)

                if key == 27:
                    stopflag = True

                # # print(key)
                # space to pause and resume
                if key == 32:
                    while True:
                        key = cv2.waitKey(1)
                        if key == 32:
                            break

                        if key == 27:
                            stopflag = True
                            break

        for eagle in eagles:
            eagle.videoSourceRelease()
        global STOP_FLAG
        STOP_FLAG = True
        cv2.destroyAllWindows()
        print('all stopped')

    def run(self):
        # getting things ready
        args = Options().parse()
        if args.subcommand is None:
            raise ValueError("ERROR: specify the experiment type")

        # run demo
        print('Runing............')
        print args
        global IS_RECORD
        if args.record == 1:
            IS_RECORD = True
        self.process(args, mirror=False)


class list_manage():
    def __init__(self):
        self.pool=[]

    def get(self):
        if self.pool.__len__()>0:
            return self.pool.pop()
        else:
            return None

    def add(self,data):
        self.pool.append(data)

    def printf(self):
        print(self.pool)

    def show(self):
        copy=self.pool[:]
        return copy

if __name__=='__main__':
    # getting things ready
    args = Options().parse()
    p = threading.Thread(target=boss,args=())
    p.start()
