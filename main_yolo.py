# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
#vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
vs = cv2.VideoCapture(0)

face_model = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
maskNet=load_model('mobilenet_v2.model')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

# loop over the frames from the video stream

while True:
    
    status , photo = vs.read()
    frame = photo
    #photo = imutils.resize(photo,width=400)

    #mask detection
    faces=face_model.detectMultiScale(photo)  

    for (x,y,w,h) in faces:
        face_img=photo[y:y+w,x:x+w]

        #face_img=cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        face_img=cv2.resize(face_img,(224,224))
        face_img=img_to_array(face_img)
        reshaped=np.reshape(face_img/255,(1,224,224,3))
        result=maskNet.predict(reshaped)
        label=0 if result[0][0]>0.8 else 1
      
        cv2.rectangle(photo,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(photo,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(photo, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
	# resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,
             personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the minimum social
            # distance
    violate = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
    if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number
                        # of pixels
                        if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
                            violate.add(i)
                            violate.add(j)

		# loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
                if i in violate:
                    color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                #cv2.imshow('frame',frame)

		# draw the total number of social distancing violations on the
		# output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
            
            

