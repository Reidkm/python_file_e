#!/usr/bin/python
# -*- coding:UTF-8 -*-
import base64


class Message(object):

    EOF = b'\n\n\n'

    def __init__(self, camera, img, detect_objs, encode=True):
        self.camera = camera
        if encode:
            self.img = base64.b64encode(img)
        else:
            self.img = img
        if type(detect_objs) != list:
            raise TypeError('detect_objs should be a list')
        self.detect_objs = detect_objs

    @staticmethod
    def from_raw_str(data):
        t = data.strip().replace('{', '').replace('}', '').split(' ')
        camera = t[0].strip().replace('[', '').replace(']', '')
        img = t[1].strip().replace('[', '').replace(']', '')
        objs = []
        for d in t[2:]:
            dt = d.strip().replace('[', '').replace(']', '').split(',')
            if len(dt) == 5:
                obj = Message.DetectObj(int(dt[0]), int(dt[1]), int(dt[2]), int(dt[3]), int(dt[4]))
                objs.append(obj)
        msg = Message(camera, img, objs, encode=False)
        return msg

    def get_transportable(self):
        return str(self) + Message.EOF

    def __str__(self):
        head = '{[%s] [%s] ' % (self.camera, self.img)
        tail = '}'
        if len(self.detect_objs) == 0:
            middle = ''
        else:
            middle = reduce(lambda x, y: '%s %s' % (str(x), str(y)), self.detect_objs)
        return '%s%s%s' % (head, middle, tail)

    class DetectObj(object):
        DETECT_OBJ_TYPE_NOTHING = 0
        DETECT_OBJ_TYPE_CONTAINER = 1
        DETECT_OBJ_TYPE_CARGO = 2

        def __init__(self, x1, y1, x2, y2, detect_obj_type=DETECT_OBJ_TYPE_NOTHING):
            self.x1 = int(x1)
            self.y1 = int(y1)
            self.x2 = int(x2)
            self.y2 = int(y2)
            self.type = detect_obj_type

        def __str__(self):
            return '[%d,%d,%d,%d,%d]' % (self.x1, self.y1, self.x2, self.y2, self.type)


if __name__ == '__main__':
    objs0 = []
    for i in range(1, 10):
        obj0 = Message.DetectObj(11, 22, 3, 44, Message.DetectObj.DETECT_OBJ_TYPE_CARGO)
        objs0.append(obj0)

    msg0 = Message(0, 'image', objs0)
    print str(msg0)
    msg1 = Message.from_raw_str(str(msg0))
    print msg1
    print msg1.get_transportable()
    # {[0] [aW1hZ2U=] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2]}
    # {[0] [aW1hZ2U=] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2]}
    # {[0] [aW1hZ2U=] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2] [11,22,3,44,2]}
    #
    #
    #

