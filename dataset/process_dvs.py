#! /usr/bin/env python

import rosbag
import rospy
import ros_numpy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import numpy.matlib
import math
import copy
import cv2

from pytictoc import TicToc
from PIL import Image
from pyproj import Proj, transform
from pathlib import Path

## global variables
t = TicToc()
width = 0
height = 0
imglist = []
cvbdg = CvBridge()
class allevents():
    def __init__(self):
        self.x = np.zeros((0,),dtype=np.uint16)
        self.y = np.zeros((0,),dtype=np.uint16)
        self.p = np.zeros((0,),dtype=np.bool)
        self.t = np.zeros((0,),dtype=np.float32)
        self.nevents = 0

def update_event_q(msg, evtlist):
    global width, height
    width = msg.width
    height = msg.height
    nevents = np.shape(msg.events)[0]
    x_arr = np.zeros((nevents,),dtype=np.uint16)
    y_arr = np.zeros((nevents,),dtype=np.uint16)
    p_arr = np.zeros((nevents,),dtype=np.bool)
    t_arr = np.zeros((nevents,),dtype=np.float64)
    for i in range(nevents):
        x_arr[i] = msg.events[i].x
        y_arr[i] = msg.events[i].y
        p_arr[i] = msg.events[i].polarity
        t_arr[i] = msg.events[i].ts.to_sec()
    order = np.argsort(t_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    p_arr = p_arr[order]
    t_arr = t_arr[order]
    evtlist.x = np.concatenate((evtlist.x, x_arr));
    evtlist.y = np.concatenate((evtlist.y, y_arr));
    evtlist.p = np.concatenate((evtlist.p, p_arr));
    evtlist.t = np.concatenate((evtlist.t, t_arr));
    evtlist.nevents += nevents
    return evtlist

def generate_event_img(evpath, evtlist, im_stamp):
    global width, height
    dt = 0.005 # 5ms
    tstamp = im_stamp
    dvs_img0 = np.zeros((height,width, 3), dtype=np.uint8)
    dvs_img1 = np.zeros((height, width, 3), dtype=np.uint8)
    dvs_img2 = np.zeros((height, width, 3), dtype=np.uint8)

    e_idx = abs(evtlist.t - tstamp)<dt
    if np.count_nonzero(e_idx) < (0.01 * height * width):
        return
    itvl = np.round(np.count_nonzero(e_idx)/3, 0)
    ev_x = evtlist.x[e_idx]
    ev_y = evtlist.y[e_idx]
    ev_p = evtlist.p[e_idx]
    ev_t = evtlist.t[e_idx]

    dvs_img0[ev_y[0*itvl:1*itvl], ev_x[0*itvl:1*itvl], ev_p[0*itvl:1*itvl]*2] = 255
    dvs_img1[ev_y[1*itvl:2*itvl], ev_x[1*itvl:2*itvl], ev_p[1*itvl:2*itvl]*2] = 255
    dvs_img2[ev_y[2*itvl:3*itvl], ev_x[2*itvl:3*itvl], ev_p[2*itvl:3*itvl]*2] = 255

    im0 = Image.fromarray(dvs_img0)
    im0.save(evpath + '0/%6.6f.png' % tstamp)
    im1 = Image.fromarray(dvs_img1)
    im1.save(evpath + '1/%6.6f.png' % tstamp)
    im2 = Image.fromarray(dvs_img2)
    im2.save(evpath + '2/%6.6f.png' % tstamp)

def main(seqname, path_prefix):
    bag = rosbag.Bag(seqname+'.bag')
    gpsfile = path_prefix + 'img_' + seqname + '/gpslist.txt'
    gpslist = open(gpsfile,"r")
    event_dir = path_prefix + 'evt_' + seqname + '/'

    image_dir = Path(path_prefix+'img_'+seqname)
    im_list = list(image_dir.glob('*.png'))
    im_stamps = sorted([float(x.split('/')[-1][:-4]) for x in im_list])
#    im_stamps = [x for i, x in enumerate(im_stamps) if np.remainder(i,5)==0] #for downsampling
    im_stamps = np.array(im_stamps)

    ## write images with timestamp assignment
    for i in range(len(im_stamps)):
        startT = rospy.Time.from_sec(im_stamps[i]-0.2)
        endT = rospy.Time.from_sec(im_stamps[i]+0.2)
        evtlist = allevents()
        for _, msg, _ in bag.read_messages(topics=['/dvs/events'], start_time=startT, end_time=endT):
            evtlist = update_event_q(msg, evtlist)
        generate_event_img(event_dir, evtlist, im_stamps[i])

    print('processed '+seqname+','+' total event images are : '+str(len(im_stamps)))

    bag.close()

if __name__ == '__main__':
    """Creates event images seperately in folders (0 [previous] / 1 [current] / 2 [next]).

    Args:
        seqname : name of bagfile.
            e.g. campus_day1, campus_day2
        path_prefix : path of output directory
            e.g. /media/data/projects/vivid/driving/
    Outputs:
        event frame images : generated event frame images
            files are saved under path_prefix/evt_/0, path_prefix/evt_/1, path_prefix/evt_/2
    """
    main(sys.argv[1], sys.argv[2])
