#!/usr/bin/env python

import rospy, math, utils, image_geometry, tf, operator
import message_filters
import tf.transformations as tf_help
import tf2_ros
import numpy as np
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, CameraInfo, PointField
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import sys
import cv2
from manager_pkg.msg import clmat
import image_geometry

NS = 'object_detector'
NODE_NAME = 'semseg_to_pcl'
PARAM_PREFIX = NS + '/' + NODE_NAME + '/'

PARAM_LIST = [	('in_pcl', 'dynamic_point_cloud'),
		('out_pcl', 'dynamic_point_cloud_clmat'),
		('in_viz', rospy.get_param('camera_dest')),
        ('image_spec',rospy.get_param('image_spec')),
		('in_semseg','semseg'),
		('max_clmat_age', 15),
		('max_pcl_age', 1500),
		('transform_wait', 2),
		('base_frame', 'base_link'),
		('map_frame', 'map'),
		('camera_frame', rospy.get_param('camera_used')),
		('image_rate', 5),
        ('clmat_rate', 5),
        ('sel_classes',rospy.get_param('sel_classes')),
		('camera_used',rospy.get_param('camera_used')),
        ('width',rospy.get_param('inputImgWidth')),#use_clmat[key].width 1616 #segnet result 360
        ('height',rospy.get_param('inputImgHeight')),#use_clmat[key].height 1232 #segnet result 480
        ('camera0PM',[638.81494, 0, 625.98561, 0, 0, 585.79797, 748.57858, 0, 0, 0, 1, 0]),
        ('sel_classes',[0, 1, 2, 3, 4, 5, 6]),
        ('colors',[[0,0,0],[255, 153, 0],[0, 255, 0],[255, 0, 255],[0,255, 255],[0, 0, 255],[0, 153, 255]])]#colors for ('Sky','Building','Pole','Road Mark','Road','Pavement','Tree','Sign','Fence','Vehicle','Pedestrian','Bike'), all clases: range(0,11)

class ToPCL:
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
        mode_sub = message_filters.Subscriber(self.in_pcl, PointCloud2)
        penalty_sub = message_filters.Subscriber(self.in_semseg + '/class_matrix', clmat)
        ts = message_filters.ApproximateTimeSynchronizer([mode_sub, penalty_sub], 200, 0.5)
        ts.registerCallback(self.callback)
        
        # topic where we publish
        self.pcl_pub = rospy.Publisher(self.out_pcl, PointCloud2, queue_size=20)
        self.pcl_pub2 = rospy.Publisher('dyn_pcl_clmat_IG', PointCloud2, queue_size=20)
        
        #setting buffer for transformation
        tfBuffer = tf2_ros.Buffer(rospy.Duration(10)) #tf buffer length
        self.tf_listener = tf.TransformListener(tfBuffer)
        self.adjustment = np.dot(tf_help.rotation_matrix(math.pi/2, (0,1,0)), tf_help.rotation_matrix(math.pi, (0,0,1)))
        print "hi"
#    def pcl_cb(self, pcl):
    def callback(self,pcl,clmats):
        if len(pcl.data) is 0:
            rospy.logwarn('No points')
            return
        if len(clmats.data) == 0:
            rospy.logwarn('There are not any segmented images ha!')
            return
        if (rospy.Time.now() - pcl.header.stamp).to_sec() > self.max_pcl_age:
            rospy.logwarn('Too old pointcloud! Skipping!')
            return
        print "ahoj"
        cloud = list(pc2.read_points(pcl, skip_nans=True))
        np_cloud = utils.cloud_to_numpy(cloud)
        #self.pcls[pcl.header.stamp] = np_cloud
        #use_pcl = self.pcls.copy()
        #pcls_frame[pcl.header.stamp] = frame_id
        #keys_pcl = sorted(use_pcl.keys(),reverse = True)
        #load point cloud
        transforms = {}
        use_clmat = {}
        #load class matrix and camera info for images
        #use_clmat = self.clmats.copy()#class matrix - clmats[tt] = vector dat z class matrix (360x480), tt - time stamp of class matrix	
        use_clmat[clmats.header.stamp] = clmats.data
        keys = sorted(use_clmat.keys(), reverse=True)#sort list of cl matrices
        use_ci = self.cis.copy()#camera info
        use_ci_keys = sorted(use_ci.keys())
        #zero matrix same size as np_cloud
        clmat_data = np.zeros((np_cloud.shape[0],)) #zero matrix same size as np_cloud		
        #transformation from base_frame to camera_frame
        for key in keys:#for each timestamp	
            if key not in use_ci_keys:#if we don't have use_ci in timestamps
                rospy.logwarn("Don't have camera info yet for image with stamp " + str(key))
                continue
            try:	
            				#self.tf_listener.waitForTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))	
                rospy.loginfo("pcl header stamp is " +str(pcl.header.stamp)+ " and clmat: " + str(key))
                #self.tf_listener.waitForTransformFull(self.camera_frame, key, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))				
                self.tf_listener.waitForTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame, rospy.Duration(self.transform_wait))				
                transforms[key] = self.tf_listener.fromTranslationRotation(*self.tf_listener.lookupTransformFull(self.camera_frame, pcl.header.stamp, pcl.header.frame_id, pcl.header.stamp, self.base_frame))
            except tf.Exception as exc:
                rospy.logwarn(exc)
                break
        for key in transforms:
        #			rospy.loginfo('Key is %s' % (key)) cis
            cam_cloud = np.dot(transforms[key], np_cloud.T).T
            cam_cloud = cam_cloud.dot(self.adjustment.T)
            print "cam cloud shape"
            print cam_cloud.shape
            cam_valid = np.where(cam_cloud[:,2] >= 0)[0]
            print "cam valid"
            print len(cam_valid)
            cam_data = cam_cloud[cam_valid]
            cam_data = cam_data.reshape((np.product(cam_data.shape)/4, 4))
            
            #print min(np.array(key) - np.array(self.timestamps))
            cam_mat = np.array(use_ci[key].P).reshape(3,4)#[np.array(key)-min(np.array(key) - np.array(self.timestamps))].P).reshape((3,4))
            #cam_mat = np.array(self.camera0PM).reshape((3,4))#projection matrix pasted in a hard way (should read adaptively from camera info, but timestamps didn't match) 
            cam_data = cam_mat.dot(cam_data.T).T
            div = cam_data[:,2]
            x = np.round(cam_data[:,0]/div)
            y = np.round(cam_data[:,1]/div)
            
            pix_x = np.where((0 <= x) & (x < self.width))[0]
            pix_y = np.where((0 <= y) & (y < self.height))[0]
            
            pix_valid = np.intersect1d(pix_x, pix_y)
            pix_coords = np.c_[x[pix_valid], y[pix_valid]]
            valid = cam_valid[pix_valid]
            coords = np.rint(pix_coords).astype(int)
            ind = coords.dot(np.array([1, self.width]).reshape((2,1)))
            Th_dataI = use_clmat[key]
            th_data = np.array(Th_dataI)
            clmat_data[valid.T] = th_data[ind]+1
            #
            th_data2 = use_clmat[key]
            cloud_dx = []
            cloud_dy = []
            cloud_dz = []
            rgb3 = []
            int4 = []
            clmat3 = []
            camera_model = image_geometry.PinholeCameraModel()
            #info = RosUtils.get_next_message("wide_stereo/left/camera_info",CameraInfo)
            camera_model.fromCameraInfo(use_ci[key])
            cam_data2 = cam_cloud[cam_valid]
            rgb2 = clmat_to_rgb(np.array(th_data2))
            rgbx, xx, yy, zz, int2 = zip(*cloud)
            int3 = [int2[i] for i in cam_valid]
            xx = list(xx[i] for i in cam_valid)
            yy = list(yy[i] for i in cam_valid)
            zz = list(zz[i] for i in cam_valid)
            clmat2 = np.array(th_data2)
            img_back = np.zeros((self.height,self.width))
            imgBGR = np.zeros((self.height,self.width,3))
            print "length and size of clmat"
            print len(clmat2)
            clmat_mat = clmat2.reshape(self.height,self.width)
            print clmat_mat.shape
            #show_img_from_clmat(self,clmat_mat)
            print np.unique(clmat_mat)
            for pix in range(cam_data.shape[0]):
                uv = camera_model.project3dToPixel((cam_data2[pix,0],cam_data2[pix,1],cam_data2[pix,2]))
                if ((uv[1] < self.height and uv[0] < self.width) and (uv[0] >= 0 and uv[1] >= 0)):
                    if (int(clmat_mat[uv[1],uv[0]]) in self.sel_classes):
                        if (int(clmat_mat[uv[1],uv[0]]) > 0):
                            # img_back[uv[0],uv[1]]=clmat2[pix]
                            imgBGR[self.height - int(uv[1])-1,self.width - int(uv[0])-1,0]=self.colors[int(clmat_mat[uv[1],uv[0]])][2]/255.
                            imgBGR[self.height - int(uv[1])-1,self.width - int(uv[0])-1,1]=self.colors[int(clmat_mat[uv[1],uv[0]])][1]/255.
                            imgBGR[self.height - int(uv[1])-1,self.width - int(uv[0])-1,2]=self.colors[int(clmat_mat[uv[1],uv[0]])][0]/255.
                            #print imgBGR[self.height - uv[1],self.width - uv[0],:]
                            cloud_dx.append(xx[pix])
                            cloud_dy.append(yy[pix])
                            cloud_dz.append(zz[pix])
                            rgb3.append(rgb2[pix])
                            int4.append(int3[pix])
                            clmat3.append(clmat_mat[uv[1],uv[0]])
            print set(clmat3)
            #imgBGR = cv2.cvtColor(imgRGB,cv2.COLOR_RGB2BGR)
            #show_img_from_clmat(self,imgBGR)
            #imgBGR = cv2.resize(imgBGR,(800,600))
            cv2.imshow('img BGR',imgBGR)
            cv2.waitKey(1)
            for cl in (set(clmat3)):
                nmb_cl = len((np.where(clmat3 == cl))[0])
                rospy.loginfo('Points per class %d: %d (valid: %d, all: %d)' % (cl, nmb_cl, len(cloud_dx), cam_data.shape[0]))
            av = range(len(cloud_dx))
            x2 = list(cloud_dx[i] for i in av)#x is tuple
            y2 = list(cloud_dy[i] for i in av)#y is tuple
            z2 = list(cloud_dz[i] for i in av)#z is tuple
            rgb4 = list(rgb3[i] for i in av)
            int5 = list(int4[i] for i in av)
            clmat4 = list(clmat3[i] for i in av)
            
            for cl_id in range(7):#0-sky, 1 - building, 2 - road, 3 - pavement, 4 - tree, 5 - vehicle, 6 - pedestrian (check which classes are selected in imageSemseg.py)
                all_valid = np.where(clmat_data == (cl_id+1))[0]
               # rospy.loginfo('For class %d found temperature for %d points out of %d' % (cl_id, all_valid.size, clmat_data.size))
            all_valid = np.where(clmat_data == self.sel_classes)[0]		
            if all_valid.size is not 0:
            			    #LOAD POINT CLOUD DATA
                clmat_field = PointField(name='clmat', offset=(pcl.fields[-1].offset + 4), datatype=PointField.FLOAT32, count=1)
                rgb, x, y, z, i = zip(*cloud)
                
  			    #SELECT VALID POINT CLOUD DATA (those which match image from camera_0 and given class)
                #get_fun = operator.itemgetter(*(all_valid.tolist()))
                x = list(x[i] for i in all_valid)#x is tuple
                y = list(y[i] for i in all_valid)#y is tuple
                z = list(z[i] for i in all_valid)#z is tuple
                			    #x = iterize(get_fun(x))
                		#y = iterize(get_fun(y))
                		#z = iterize(get_fun(z))
                			#THIS HAS TO BE COMMENTED - IT TRANSFORMS POINT CLOUD DATA TO MAP COORDINATES
                		#points = np.array(zip(x, y, z, np.ones(np.shape(x))))
                		#try:
                	#	self.tf_listener.waitForTransform(self.map_frame, pcl.header.frame_id, pcl.header.stamp, rospy.Duration(self.transform_wait))
                	#	transform = self.tf_listener.fromTranslationRotation(*self.tf_listener.lookupTransform(self.map_frame, pcl.header.frame_id, pcl.header.stamp))
                		#except tf.Exception as exc:
                	#	rospy.logwarn(exc)
                	#	return
                		#points = np.dot(transform, points.T).T
                		#x, y, z, w = zip(*points)
                			    #COLOR DATA BASED ON CLASS IT HAS
                clmat = clmat_data[all_valid]
                rgb = clmat_to_rgb(clmat)
                data = zip(rgb, x, y, z, i, clmat)
                msg = pc2.create_cloud(pcl.header, pcl.fields + [clmat_field], data)
                #			    data = zip(x, y, z)
                #			    msg = pc2.create_cloud(pcl.header, pcl.fields[1:4], data)
                print "len x2"
                print len(x2)
                self.pcl_pub.publish(msg)
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

def show_img_from_clmat(self,clmat_mat):	
    imgBGR = np.zeros((self.height,self.width,3))
    for i in range(7):
        [rows,cols] = np.where(clmat_mat == i)
        imgBGR[rows,cols,0]=self.colors[i][2]/255
        imgBGR[rows,cols,1]=self.colors[i][1]/255
        imgBGR[rows,cols,2]=self.colors[i][0]/255
    cv2.imshow('2D image from clmat',imgBGR)
    cv2.waitKey(1)

if __name__ == '__main__':
	pcl = ToPCL()
	rospy.spin()
