#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author: Karla Stepanova, CIIRC CVUT, 10.2.2017
#ros node for semantics segmentation of incoming data from TRADr robot - based on SegNet implementation
#subscribes to input image from robot camera, segment them and show the resulting image
#SET: on line 47 change self.categ to individualize which categories you want to segment (all are range(0,11))
#    on line 48 define self.timePerClass to set the time per processing one class -> so we process next image after the ones before were processed - this should be automated to image timestamps
#TODO: should be quicker, process images according to speed so it is fluent
#    publish area and description of each class to further processing in 3d point cloud and saved in database
#http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
#

import rospy
from sensor_msgs.msg import CompressedImage, Image
#from std_msgs.msg import Int32MultiArray, MultiArrayDimension
import cv2
import numpy as np
import sys, time
import imutils
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from manager_pkg.msg import clmat
from scipy.misc import imresize
from skimage.transform import resize

import threading
from Queue import Queue, PriorityQueue

VERBOSE = False
TIMIT = False
NS = 'semseg'
NODE_NAME = 'imageSemseg'
PARAM_PREFIX = NS + '/' + NODE_NAME + '/'

PARAM_LIST = [('out_semseg','semseg'),
        ('out_clmat','class_matrix'),
        ('out_imsemseg','image_semseg'),
        ('out_contours','image_contours'),
        ('image_spec',rospy.get_param('image_spec')),
        ('camera_dest',rospy.get_param('camera_dest')),
        ('camera_used',rospy.get_param('camera_used')),
        ('width',rospy.get_param('inputImgWidth')),
        ('height',rospy.get_param('inputImgHeight')),
        ('segnet_width',480),
        ('segnet_height',360),
        ('categ',rospy.get_param('used_classes')),##defines categories to use ('Sky','Building','Pole','Road Mark','Road','Pavement','Tree','Sign','Fence','Vehicle','Pedestrian','Bike'), all clases: range(0,11)
        ('timePerClass',200),#defines the time per class, default 1100
        ('classesSN', ['Sky','Building','Pole','Road Mark','Road','Pavement','Tree','Sign','Fence','Vehicle','Pedestrian','Bike']),
        ('colors',[[0,0,0],[255, 153, 0],[0, 255, 0],[255, 0, 255],[0,255, 255],[0, 0, 255],[0, 153, 255]])]#colors for ('Sky','Building','Pole','Road Mark','Road','Pavement','Tree','Sign','Fence','Vehicle','Pedestrian','Bike'), all clases: range(0,11)

class ImageSemseg:

    def __init__(self):
        #initialization of SegNet network
        sys.path.append('/usr/local/lib/python2.7/site-packages')
        # Make sure that caffe is on the python path:
        caffe_root = '/home/tradr/Documents/caffe-segnet/'
        sys.path.insert(0, caffe_root + 'python')
        import caffe

        #import parameters to self
        for name, default in PARAM_LIST:
            setattr(self, name, rospy.get_param(PARAM_PREFIX + name, default))

        # Import arguments
        model = '/home/tradr/Documents/SegNet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt'
        weights = '/home/tradr/Documents/SegNet-Tutorial/Example_Models/segnet_weights_driving_webdemo.caffemodel'

        #network learning
        self.net = caffe.Net(model,
                    weights,
                    caffe.TEST)
        caffe.set_mode_gpu()
        self.imgNmb = 1

        # subscribed Topic
            #self.subscriber = rospy.Subscriber("/viz/camera_0/image/compressed",
            #    CompressedImage, self.callback,  queue_size = 1)
        self.subscriber = rospy.Subscriber("/"+self.camera_dest + "/"+self.camera_used+ "/" +self.image_spec, CompressedImage, self.callback,  queue_size = 1)#image from camera_used
        if VERBOSE :
            print "subscribed to /"+self.camera_dest+"/"+self.camera_used+"/"+self.image_spec

        # topic where we publish
        self.image_pub = rospy.Publisher(self.out_semseg + '/' + self.out_contours, CompressedImage, queue_size = 1)#segmented image with contours
        self.image_pub2 = rospy.Publisher(self.out_semseg + '/' + self.out_imsemseg, CompressedImage, queue_size = 1)#segmented image
        self.image_pub3 = rospy.Publisher(self.out_semseg + '/' + self.out_clmat, clmat, queue_size = 1)#class matrix from segmented image

        self.duration = rospy.Duration(0)
        self.lastTimestamp = rospy.Time(0)
       
        colours = '/home/tradr/Documents/SegNet-Tutorial/Scripts/camvid12.png'
        self.label_colours = cv2.imread(colours).astype(np.uint8)

    def normalized(self, rgb):
	    #return rgb/255.0
	    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

	    b=rgb[:,:,0]
	    g=rgb[:,:,1]
	    r=rgb[:,:,2]      


	    norm[:,:,0]=cv2.equalizeHist(b)
	    norm[:,:,1]=cv2.equalizeHist(g)
	    norm[:,:,2]=cv2.equalizeHist(r)

	    return norm

    def callback(self, ros_data):
        #time stamping so new images are processed only after the old ones are finished
        #frequency of segmented images will be therefore lower than the image input frequency
        #if ((time.time()-self.time)*1000 < (self.timePerClass*len(self.categ))):
    #    return

        #save timestamp of image to pass it together with segmented image
        self.timestamp = ros_data.header.stamp
        classesSN = ['Sky','Building','Pole','Road Mark','Road','Pavement','Tree','Sign','Fence','Vehicle','Pedestrian','Bike']

        if self.timestamp - self.lastTimestamp < self.duration:
            return
        self.lastTimestamp = self.timestamp
        computationStart = rospy.get_rostime()

        self.imgNmb = self.imgNmb+1
        start1 = time.time()
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format
        rawImdata = np.fromstring(ros_data.data, np.uint8)
        bgrRaw = cv2.imdecode(rawImdata, cv2.IMREAD_COLOR)
        #bgrRaw = self.normalized(bgrRaw)
        # rgbRaw = cv2.cvtColor(bgrRaw, cv2.COLOR_BGR2RGB)
        input_shape = self.net.blobs['data'].data.shape
        image = cv2.resize(bgrRaw, (input_shape[3],input_shape[2]))
        input_image = image.transpose((2,0,1))
        input_image = np.asarray([input_image])
        out = self.net.forward_all(data=input_image)
        end1 = time.time()
        print '%30s' % 'Executed SegNet in ', str((end1 - start1)*1000), 'ms'

        start1 = time.time()
        start = time.time()
        segmentation_ind = np.squeeze(self.net.blobs['argmax'].data)
        segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
        segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
        if TIMIT:
            end = time.time()
            print '%30s' % 'Executed Stage 1 in ', str((end - start)*1000), 'ms'
            start = time.time()

        opening2 = []
        probabilities = []
        #segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
        end = time.time()

        inpQue = Queue(len(self.categ))
        outQue = PriorityQueue(len(self.categ))
        probQue = PriorityQueue(len(self.categ))

        for i in self.categ:
            inpQue.put_nowait(i)

        # create threads
        for i in self.categ:
            thread = threading.Thread(target=morph, args=(inpQue, outQue, probQue, segmentation_ind, self.label_colours, classesSN, image))
            thread.start()

        # wait until all threads are finished and copy the results to the list 'openning2'
        inpQue.join()

        if outQue.empty():
            rospy.logwarn('Class image smoothing faulted (empty queue returned)!')
            return
        while not outQue.empty():
            item = outQue.get()
            opening2.append(item[1])
            prob = probQue.get()
            probabilities.append(prob[1])

        opening = opening2[0]
        if TIMIT:
            end = time.time()
            print '%30s' % 'Executed Stage 2 in ', str((end - start)*1000), 'ms'
            start = time.time()
        # show the image
        #cv2.imshow('cv_img',image)
        #cv2.imwrite('imgs/all3/all'+str(self.imgNmb)+'.png',image)
        #cv2.waitKey(2)

        #MERGING THE GENERATED ONE-CLASS IMAGES TOGETHER
        classImage = np.empty([opening.shape[0]*opening.shape[1]+1])
        classImageN = np.empty([self.width*self.height])
        classImageN2 = np.empty([self.width*self.height])
        vectorizedImageSize = opening2[0].shape[0] * opening2[0].shape[1]
        # maxClasses contains per-pixel info about the maximal class in that pixel; has vectorized form
        maxClasses = np.reshape(np.argmax(probabilities, 0), (vectorizedImageSize, ))
        # vectorOpenings is the openings2 matrix with the pixel coordinates flattened (i.e. [class, color, flattenedCoords])
        vectorOpenings = np.reshape(opening2, (len(self.categ), vectorizedImageSize, 3)).transpose(0, 2, 1)
        # openingMerged contains the color of the "maximal" class for each pixel
        openingMerged = np.reshape(np.transpose(np.choose(maxClasses, vectorOpenings), (1, 0)), (opening2[0].shape[0], opening2[0].shape[1], 3))
        classImage = maxClasses + 1
        classImage[~np.any(np.reshape(probabilities, (len(self.categ), vectorizedImageSize)), 0)] = 0

        cv2.imwrite('imgs/all5/Mall'+str(self.imgNmb)+'_el.png',openingMerged*255)
        cv2.imshow('mergedImage',openingMerged)
        cv2.waitKey(1)
        if TIMIT:
            end = time.time()
            print '%30s' % 'Executed Stage 3 in ', str((end - start)*1000), 'ms'
            start = time.time()

        #WE HAVE TO RESIZE SEGMENTED IMAGE BACK SO IT MATCHES 3D POINT CLOUD DATA
        #reshape classImage to original size of the input image - TODO automate height and width
#        data = classImage[1:classImage.shape[0]]
#        data = data.reshape(360, 480)
        data = classImage.reshape(self.segnet_height, self.segnet_width)
        #show_img_from_clmat(self,data)        
        b = data.copy()
        b = imresize(b, (self.height,self.width), interp='nearest', mode='F')
        #b = cv2.resize(b, (self.height,self.width))
        #show resized clmat 
        #show_img_from_clmat(self,b)        
        b = b.reshape(1, self.width*self.height)
        classImageN[0:self.width*self.height] = b
        classImage2 =np.array(classImageN, dtype = np.float32)
        # PUBLISH DATA
        #Publish new image with contours
        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = np.array(cv2.imencode('.jpg',image)[1]).tostring()
        self.image_pub.publish(msg)
        #Publish segmented image
        msg2 = CompressedImage()
        msg2.data = openingMerged*255
        self.image_pub2.publish(msg2)
        #Publish array with class nmbs       
        msg3 = clmat()
        msg3.header.stamp = self.timestamp
        msg3.data = classImage2
        self.image_pub3.publish(msg3)
        #classImageN[0] = str(self.timestamp)
        #classImageN[1:self.width*self.height+1] = b
        #classImage2 =np.array(classImageN, dtype = np.float32)
        #self.image_pub3.publish(classImage2)
        if TIMIT:
            end = time.time()
            print '%30s' % 'Executed Stage 4 in ', str((end - start)*1000), 'ms'
        end1 = time.time()
        print '%30s' % 'Executed whole segmentation in ', str((end1 - start1)*1000), 'ms'
        self.duration = rospy.get_rostime() - computationStart

def morph(inpQue, outQue, probQue, segmentation_ind, label_colours, classesSN, image):
    #for each colour create BW image, then morphologically close it and convert
    #to the appropriate color of a given class. This will ensure that in the
    #process of morphological closing will not be created new colors which do
    #not correspond to any class
    try:
        ind = inpQue.get()
        segmentation = np.zeros(segmentation_ind.shape, dtype=np.float)
        segmentation_rgb = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.float)

        #ind-th class is to be white
        segmentation[np.where(segmentation_ind == ind)] = 1

        #morphological closing and opening of image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closing = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        #CONVERT THE BW IMAGE BACK TO COLOUR
        choose_idx = np.where(opening)
        segmentation_rgb[choose_idx[0], choose_idx[1], :] = label_colours[0, ind].astype(float)/255

        # should be substituted by some class presence probability or something, for now, this will do (i.e. color based "probability")
        summmedOpening = np.sum(segmentation_rgb, 2)
        openingProb = summmedOpening #/ np.max(summmedOpening)

        outQue.put((ind, segmentation_rgb))
        probQue.put((ind, openingProb))
        #WE HAVE TO RESIZE SEGMENTED IMAGE BACK SO IT MATCHES 3D POINT CLOUD DATA
    #    opening = cv2.resize(opening, (bgrRaw.shape[1],bgrRaw.shape[0]))#cv2.resize is terribly slow -> changed to np.resize on the segmented image
    except:
        rospy.logerr('There was an exception while applying morphological operations to segmented images')
    finally:
        inpQue.task_done()
    return

def show_img_from_clmat(self,clmat_mat):	
    imgBGR = np.zeros((self.height,self.width,3))
    for i in range(7):
        [rows,cols] = np.where(clmat_mat == i)
        imgBGR[rows,cols,0]=self.colors[i][2]/255
        imgBGR[rows,cols,1]=self.colors[i][1]/255
        imgBGR[rows,cols,2]=self.colors[i][0]/255
    cv2.imshow('2D image from clmat',imgBGR)
    cv2.waitKey(1)

def main():
    ic = ImageSemseg()
    rospy.init_node('image_semseg', anonymous=True)
   # rospy.on_shutdown(ic.shutdown)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image semantic segmentation module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
