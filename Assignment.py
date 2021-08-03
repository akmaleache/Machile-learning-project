# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import pymongo
import dns.resolver
import dns.exception
import os

# mycol.drop()
#from google.colab.patches import cv2_imshow

net = cv2.dnn.readNet('C:/Users/Akmal/ml_assignment/yolov3.weights', 'C:/Users/Akmal/ml_assignment/yolov3.cfg')
classes = []

with open('C:/Users/Akmal/ml_assignment/coco.names') as f:
    classes = f.read().splitlines()
total_images = len([i for i in os.listdir('C:/Users/Akmal/ml_assignment/images/')])

### connection to mongodb
    
client = pymongo.MongoClient("mongodb://localhost:27017/")
    
mydb = client['Assignment']
mycol = mydb['Image_classes']



for i in os.listdir('C:/Users/Akmal/ml_assignment/images/'):
    print(i)
    
    img = cv2.imread('C:/Users/Akmal/ml_assignment/images/'+i)
    height, width, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)
    
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n),img_blob)
    #         cv2.waitKey(1000)
            
    
    net.setInput(blob)
    
    outputLN = net.getUnconnectedOutLayersNames()
    layerO = net.forward(outputLN)
    
    box = list()
    confidences = list()
    class_ids = list()
    
    for output in layerO:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >0.5:
                c_x = int(detect[0]* width)
                c_y = int(detect[1]* height)
                w = int(detect[2]* width)
                h = int(detect[3]* height)
    
                x = int(c_x - w/2)
                y = int(c_x - h/2)
                
                box.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes  = cv2.dnn.NMSBoxes(box, confidences,0.5,0.4)
    print(indexes.flatten())
    
    font = cv2.FONT_HERSHEY_PLAIN
    cols = np.random.uniform(0,255,size = (len(box),3))
    store = list()
    
    for j in indexes.flatten():
        x,y,w,h = box[j]
        lable = str(classes[class_ids[j]])
        store.append(lable)
        confidence = str(round(confidences[j]))
        color = cols[j]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img,lable + " " + confidence, (x,y+20), font, 2, (255,255,255),2 )
        
    # cv2.imwrite('C:/Users/Akmal/ml_assignment/saved_img/'+img_name,img)
    # cv2.waitKey(10)
    mydict= {'image_name':i,'classes':store}
    x = mycol.insert_one(mydict)
   

obj = input('Enter Object ')

q = {'classes':obj}
mydoc =  mycol.find(q)
for x in mydoc:  
    img = cv2.imread('C:/Users/Akmal/ml_assignment/images/'+x['image_name'])
    cv2.imshow(obj,img)
    cv2.waitKey(1000)
    
    
    
    
# here it showed configuration error while uploading image in mongobd, which does not get resolved.

## NEXT STEP: after saving images in mongodb, i will use input text where user enter
## what they want to search then the input is compared with lable saved in mongo and return image cluster that have input lable.
 # mycol.drop()
