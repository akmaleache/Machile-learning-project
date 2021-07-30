# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
#from google.colab.patches import cv2_imshow

net = cv2.dnn.readNet('C:/Users/Akmal/ml_assignment/yolov3.weights', 'C:/Users/Akmal/ml_assignment/yolov3.cfg')
classes = []

with open('C:/Users/Akmal/ml_assignment/coco.names') as f:
    classes = f.read().splitlines()

len(classes)
img_name = 'img5.jpg'

img = cv2.imread('C:/Users/Akmal/ml_assignment/images/'+img_name)
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

for i in indexes.flatten():
    x,y,w,h = box[i]
    lable = str(classes[class_ids[i]])
    confidence = str(round(confidences[i]))
    color = cols[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img,lable + " " + confidence, (x,y+20), font, 2, (255,255,255),2 )
    
cv2.imwrite('C:/Users/Akmal/ml_assignment/saved_img/'+img_name,img)
# cv2.waitKey(10)

### connection to mongodb

# import pymongo
# import dns.resolver
# import dns.exception
# import dns
# import urllib.parse


# my_client = pymongo.MongoClient('mongodb+srv://AkmalEache:'+urllib.parse.quote_plus('akmal@786')+'@cluster0.zubrn.mongodb.net/Assignment?retryWrites=true&w=majority')

# mydb = my_client['Assignment']
# mycol = mydb['Image_classes']

# mydict= {'image':img,'classes':lable}
# mycol.insert_one(mydict)

# here it showed configuration error while uploading image in mongobd, which does not get resolved.

## NEXT STEP: after saving images in mongodb, i will use input text where user enter
## what they want to search then the input is compared with lable saved in mongo and return image cluster that have input lable.

