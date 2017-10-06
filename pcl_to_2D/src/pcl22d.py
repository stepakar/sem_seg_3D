#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 09:40:15 2017

@author: karla stepanova
"""

#!/usr/bin/env python
import rospy, math, image_geometry, tf,utils
import tf.transformations as tf_help
import tf2_ros
import numpy as np
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, CameraInfo, PointField
import sys
import cv2

NS = 'object_detector'
NODE_NAME = 'semseg_to_pcl'
PARAM_PREFIX = NS + '/' + NODE_NAME + '/'

PARAM_LIST = [	('in_pcl', 'dynamic_point_cloud'),
		('out_img', 'image_from_pcl'),
		('in_viz', 'viz'),
        ('image_spec',"image/compressed"),
		('in_semseg','semseg'),
		('max_pcl_age', 1500),
		('transform_wait', 2),
		('base_frame', 'base_link'),
		('map_frame', 'map'),
		('camera_frame', 'camera_1'),
		('image_rate', 5),
		('camera_used','camera_1'),
        ('width',1232),#use_clmat[key].width 1616 #segnet result 360
        ('height',1616),#use_clmat[key].height 1232 #segnet result 480
        ('camera0PM',[638.81494, 0, 625.98561, 0, 0, 585.79797, 748.57858, 0, 0, 0, 1, 0]),
        ('colors',[[50,50,50],[255, 153, 0],[0, 255, 0],[255, 0, 255],[255, 0, 0],[0, 255, 255],[0, 0, 255]])]

class PCLtoImage:
    def __init__(self):
        rospy.init_node(NODE_NAME)
        sys.path.append('~/Documents/RobotSemantics/semseg_ws/src/image_segnet/object_class_pcl/src')
        
        #import parameters to self		
        for name, default in PARAM_LIST:
            setattr(self, name, rospy.get_param(PARAM_PREFIX + name, default))
        self.clmats = {}
        self.cis = {}
        self.imgsC = {}
        self.timestamps = []
        self.counterTS = 0
        self.counter = 0
        self.clmat_counter = 0
        self.pcls = {}
        
        # subscribed Topics
        #rospy.Subscriber(self.in_pcl, PointCloud2, self.pcl_cb)
        rospy.Subscriber(self.in_viz + '/'+self.camera_used+'/camera_info', CameraInfo, self.ci_cb)
        rospy.Subscriber(self.in_viz + '/'+self.camera_used+'/'+self.image_spec, CompressedImage , self.image_cb)
        #rospy.Subscriber(self.in_semseg + '/class_matrix', clmat, self.clmat_cb)
        #approximate time synchranization of subscribed topics
        rospy.Subscriber(self.in_pcl, PointCloud2, self.callback)      
        # topic where we publish
        self.pcl_pub2 = rospy.Publisher('img_from_pcl', PointCloud2, queue_size=20)
        
        #setting buffer for transformation
        tfBuffer = tf2_ros.Buffer(rospy.Duration(10)) #tf buffer length
        self.tf_listener = tf.TransformListener(tfBuffer)
        self.adjustment = np.dot(tf_help.rotation_matrix(math.pi/2, (0,1,0)), tf_help.rotation_matrix(math.pi, (0,0,1)))
        print "hi"
        #    def pcl_cb(self, pcl):
    def callback(self,pcl):
        if len(pcl.data) is 0:
            rospy.logwarn('No points')
            return
        cloud = list(pc2.read_points(pcl, skip_nans=True))
        np_cloud = utils.cloud_to_numpy(cloud)
        transforms = {}
        #load class matrix and camera info for images
        #use_clmat = self.clmats.copy()#class matrix - clmats[tt] = vector dat z class matrix (360x480), tt - time stamp of class matrix	
        use_ci = self.cis.copy()#camera info
        use_ci_keys = sorted(use_ci.keys())
        #zero matrix same size as np_cloud
        
        #self.tf_listener.waitForTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))	
        rospy.loginfo("pcl header stamp is " +str(pcl.header.stamp))
        #self.tf_listener.waitForTransformFull(self.camera_frame, key, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))				
        self.tf_listener.waitForTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))				
        transforms = self.tf_listener.fromTranslationRotation(*self.tf_listener.lookupTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame))
        cam_cloud = np.dot(transforms, np_cloud.T).T
        cam_cloud = cam_cloud.dot(self.adjustment.T)            
        cam_valid = np.where(cam_cloud[:,2] >= 0)[0]
        print "cam valid"
        print len(cam_valid)
        cam_data = cam_cloud[cam_valid]
        cam_data = cam_data.reshape((np.product(cam_data.shape)/4, 4))
        cloud_dx = []
        cloud_dy = []
        cloud_dz = []
        rgb3 = []
        int4 = []
        camera_model = image_geometry.PinholeCameraModel()
        #info = RosUtils.get_next_message("wide_stereo/left/camera_info",CameraInfo)
        camera_model.fromCameraInfo(use_ci[use_ci_keys[0]])
        cam_data2 = cam_cloud[cam_valid]
        rgbx, xx, yy, zz, int2 = zip(*cloud)
        int3 = [int2[i] for i in cam_valid]
        xx = list(xx[i] for i in cam_valid)
        yy = list(yy[i] for i in cam_valid)
        zz = list(zz[i] for i in cam_valid)
        img_back = np.zeros((self.height,self.width))
        imgBGR = np.zeros((self.height,self.width,3))
        for pix in range(cam_data.shape[0]):
            uv = camera_model.project3dToPixel((cam_data2[pix,0],cam_data2[pix,1],cam_data2[pix,2]))
            if ((uv[1] < self.height and uv[0] < self.width) and (uv[0] >= 0 and uv[1] >= 0)):
               # img_back[uv[0],uv[1]]=clmat2[pix]
                imgBGR[self.height - uv[1],self.width - uv[0],0]=self.colors[0][2]
                imgBGR[self.height - uv[1],self.width - uv[0],1]=self.colors[0][1]
                imgBGR[self.height - uv[1],self.width - uv[0],2]=self.colors[0][0]
                cloud_dx.append(xx[pix])
                cloud_dy.append(yy[pix])
                cloud_dz.append(zz[pix])
                rgb3.append(rgbx[pix])
                int4.append(int3[pix])
        #imgBGR = cv2.cvtColor(imgRGB,cv2.COLOR_RGB2BGR)
        show_img_from_clmat(self,imgBGR)
        av = range(len(cloud_dx))
        x2 = list(cloud_dx[i] for i in av)#x is tuple
        y2 = list(cloud_dy[i] for i in av)#y is tuple
        z2 = list(cloud_dz[i] for i in av)#z is tuple
        rgb4 = list(rgb3[i] for i in av)
        int5 = list(int4[i] for i in av)
        #publish pcl from image_geometry
        data2 = zip(rgb4, x2, y2 , z2, int5)
        msg = pc2.create_cloud(pcl.header, pcl.fields, data2)
        self.pcl_pub2.publish(msg)
    def ci_cb(self, ci):
        self.cis[ci.header.stamp] = ci	
        self.timestamps.append(ci.header.stamp)
#		self.counterTS += 1	
		#self.rm_old(self.cis)

    def rm_old(self, data):#remove old images
        now = rospy.Time.now()
        rm = [key for key in data if (float(str(now)) - float(str(key)))/10000000000 > self.max_clmat_age]
        for key in rm:
            del data[key]

    def image_cb(self, image):#saves coming images
        if self.counter is 0:
            self.imgsC[image.header.stamp] = image
     #       self.rm_old(self.imgsC)
        self.counter += 1
        self.counter %= self.image_rate

    def clmat_cb(self, data):#saves coming class matrices
        rospy.loginfo( "New class mat from semseg node")
        if self.clmat_counter is 0:
            #tt = rospy.Time((data.data[0]/1000000000))
            rospy.loginfo( "clmat counter 0")
            #self.clmats[tt] = data.data[1:data.data.shape[0]]
        self.clmats[data.header.stamp] = data.data
			#self.rm_old(self.clmats)
        rospy.loginfo( "saving new data...")
        self.clmat_counter += 1
        self.clmat_counter %= self.clmat_rate
		
def clmat_to_rgb(clmat):
    min_clmat = np.min(clmat)
    max_clmat = np.max(clmat)
    rospy.loginfo('Min and max temperature: %.2f, %.2f' % (min_clmat, max_clmat))
    scaled = ((clmat - min_clmat)/(max_clmat - min_clmat))
    z = np.zeros(clmat.shape)
    o = np.ones(clmat.shape)
    r = np.minimum(np.maximum(z, 1.5 - abs(1 - 4*(scaled-0.5))),o)
    g = np.minimum(np.maximum(z, 1.5 - abs(1 - 4*(scaled-0.5))),o)
    b = np.minimum(np.maximum(z, 1.5 - abs(1 - 4*(scaled-0.5))),o)    
    #r = np.minimum(np.maximum(z, rgb_array[1,clmat]),o)
    #g = np.minimum(np.maximum(z, rgb_array[2,clmat]),o)
    #b = np.minimum(np.maximum(z, rgb_array[3,clmat]),o)
    mat = (256*256*256 - 1) * r + (256*256 - 1) * g + 255 * b
    return mat

def show_img_from_clmat(self,img):	
    self.colors   
    #imgRGB = np.zeros((self.height,self.width,3))
    #for i in range(7):
        #imgRGB[np.where(img == i)[0]] == self.colors[i]
    #print imgRGB.shape
    cv2.imshow('2D image from clmat',img)
    cv2.waitKey(1)

if __name__ == '__main__':
    pcl = PCLtoImage()
    rospy.spin()
