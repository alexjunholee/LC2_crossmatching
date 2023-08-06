#! /usr/bin/env python

import rosbag
import rospy
import ros_numpy
import sys
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import numpy.matlib
import math
import copy
import cv2
from PIL import Image
from pyproj import Proj, transform

## global variables
imgno = 0
cvbdg = CvBridge()
proj_UTMK = Proj(init='epsg:5178')
proj_WGS84 = Proj(init='epsg:4326')
initpt = np.array([987800., 1818500.])

DIM = (1280, 720)
K_1 = np.array([[930.2304004161798, 0., 643.8181772305601], [0., 934.8188524214474, 362.86281040664767], [0., 0., 1.]])
D_1 = np.array([[0.10017371941383502, -0.16488081864429036, -0.0026228370243790516, 0.002398704902301937]])

#    cv_img = (cv_img/50).astype('uint8')
#    DIM = (640, 512)
#    K_1 = np.array([[437.40300321542014, 0., 312.1060828078799], [0., 438.3160668918748, 250.70382767459375], [0., 0., 1.]])
#    D_1 = np.array([[0.248994091117758, -2.909390816436668, 8.052903070857168, -7.363581411738435]])
map1_1, map2_1 = cv2.initUndistortRectifyMap(K_1, D_1, np.eye(3), K_1, DIM, cv2.CV_16SC2)

def write_gps(msg, f_gps):
    global proj_UTMK, proj_WGS84
    tnow = msg.header.stamp.to_sec()
    lat = msg.latitude
    lon = msg.longitude
    x, y = transform(proj_WGS84,proj_UTMK,lon,lat)
    f_gps.write('%.6f, %.6f, %.6f\n'%(x-initpt[0],y-initpt[1],tnow))

def write_images(msg, folder):
    global f, imgno, cvbdg, map1_1, map2_1
    imgno += 1
    if (imgno % 10) != 0:
        return
    tnow = msg.header.stamp.to_sec()
    cv_img = cvbdg.compressed_imgmsg_to_cv2(msg,desired_encoding="passthrough")
    cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
    
    undistorted_img = cv2.remap(cv_img, map1_1, map2_1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    resized_img = cv2.resize(undistorted_img,(1280,720))
    cv2.imwrite(folder+'/images/image%05d.png'%(imgno), resized_img)
    with open(folder+'/imglist.txt', "a") as f:
        f.write("%05d %.6f\n"%(imgno, tnow))

def raw_to_kelvin(val):
    return ( 1428.0 / log( 408825.0 / ( val + 58.417 ) + 1.0 ) )

def main(seqname, folder):
    bag = rosbag.Bag(seqname+'.bag')
    folder = "data/" + folder
    os.mkdir(folder)
    os.mkdir(folder+'/img')
    f_gps = open(folder + '/gpslist.txt','w')

    # write images with timestamp assignment
    for topic, msg, t in bag.read_messages(topics=['/gps']):
        write_gps(msg, f_gps)
    print('saved '+str(folder)+' gps timstamps.')

    for topic, msg, t in bag.read_messages(topics=['/rs_front_camera1/color/image_raw/compressed']):
#    for topic, msg, t in bag.read_messages(topics=['/thermal/image_raw']):
        write_images(msg, folder)
        if imgno % 1000 == 0:
            print('saved '+str(imgno)+' images timstamps to detections.')

    bag.close()

if __name__ == '__main__':
    """Saves images and its GPS (UTM) coordinates from Visibility dataset.

    Args:
        seqname : name of bagfile.
            e.g. campus_day1, campus_day2
        folder : path of output directory
            e.g. /media/data/projects/vivid/driving/
    Outputs:
    	gpslist.txt : position from GPS in UTM coordinates (origin set from custom position)
    	    formatted in "utm_x utm_y timestamp".
        images: files are saved under folder/img.
    """
    main(sys.argv[1],sys.argv[2])
