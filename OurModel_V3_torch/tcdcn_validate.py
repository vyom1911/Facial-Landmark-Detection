import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch.autograd as A
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import math
from tcdcnn_dataset import FaceLandmarksDataset
import torch.nn.functional as F
import logging
import sys
from os.path import isfile, join
from tcdcn_final import TCDCNN
logging.basicConfig(filename='example.log',level=logging.DEBUG)

mypath = "/home/vyom/ADA_Project/MTFL/"
train= "training.txt"
test= "testing.txt"
val = "validate.txt"
val_anno = "validation_bb.txt"
annotation = "annotation.txt"
us="us.txt"
us_anno = "annotation_testing_us.txt"
faceLandMark = FaceLandmarksDataset(mypath,val,val_anno)
dataloader = DataLoader(faceLandMark, batch_size=64,
                        shuffle=False, num_workers=4)

#i,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,g,s,gl,p = np.genfromtxt(join(mypath,val), delimiter=" ", unpack=True)
i= np.genfromtxt(join(mypath, val), delimiter=" ",usecols=0, dtype=str, unpack=True)
bb1,bb2,bb3,bb4, bProb = np.genfromtxt(join(mypath,val_anno), delimiter=" ", unpack=True)

#Converting Annotation according to resized images
ratio_x=40/(bb3-bb1)
ratio_y=40/(bb4-bb2)
#l1,l2,l3,l4,l5,l6,l7,l8,l9,l10 = (l1-bb1)*ratio_x,(l2-bb1)*ratio_x,(l3-bb1)*ratio_x,(l4-bb1)*ratio_x,(l5-bb1)*ratio_x,(l6-bb2)*ratio_y,(l7-bb2)*ratio_y,(l8-bb2)*ratio_y,(l9-bb2)*ratio_y,(l10-bb2)*ratio_y

i = [k.replace('\\','/') for k in i]

onlyfiles = [ f for f in i if isfile(join(mypath,f))]
File_length=len(onlyfiles)
out_images = list()
indexes = list()
for n in range(0, File_length):
    #temp = cv2.resize(cv2.imread( join(mypath, onlyfiles[n]),0),(40,40))
    temp = cv2.imread(join(mypath, onlyfiles[n]))
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp.astype(np.uint8)
    out_images.append(temp)
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    crop_img = gray[int(bb2[n]):int(bb4[n]), int(bb1[n]):int(bb3[n])]
    if(crop_img.shape[0]<40 or crop_img.shape[1]<40):
        indexes.append(n)
        continue
    resized = cv2.resize(crop_img,(40,40),interpolation=cv2.INTER_AREA)

model = torch.load('/home/vyom/ADA_Project/OurModel_V3_torch/tcdcn.pt')
prediction=list()
genderList=list()
glassesList=list()
smileList = list()
poseList=list()
for i,data in  enumerate(dataloader,1):  
    images,landmark,gender,smile,glass,pose = data
    images  = A.Variable(images)
    image = images.data.cpu().numpy()	
    images_temp = images.clone()
    landmark = A.Variable(landmark)
    x_one,x_two,x_three,x_four,x_five = model(images.float()) #prediction code
    predict=x_one.data.cpu().numpy()
    #print(len(images))
    prediction.append(predict)
    genderList.append(x_two.data.cpu().numpy())
    smileList.append(x_three.data.cpu().numpy())
    glassesList.append(x_four.data.cpu().numpy())
    poseList.append(x_five.data.cpu().numpy())

print("prediciton: ",predict[-1])
print("actual ",landmark[-1,0:10])

#Converting predicted landmarks back to original images
ratio_x=abs(bb3-bb1)/40
ratio_y=abs(bb4-bb2)/40
nl1,nl2,nl3,nl4,nl5,nl6,nl7,nl8,nl9,nl10=list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
for predicted in prediction:
    for n in range(len(predicted)):
        nl1.append((predicted[n]*ratio_x[n])+bb1[n])
        nl2.append((predicted[n]*ratio_x[n])+bb1[n])
        nl3.append((predicted[n]*ratio_x[n])+bb1[n])
        nl4.append((predicted[n]*ratio_x[n])+bb1[n])
        nl5.append((predicted[n]*ratio_x[n])+bb1[n])
        nl6.append((predicted[n]*ratio_y[n])+bb2[n])
        nl7.append((predicted[n]*ratio_y[n])+bb2[n])
        nl8.append((predicted[n]*ratio_y[n])+bb2[n])
        nl9.append((predicted[n]*ratio_y[n])+bb2[n])
        nl10.append((predicted[n]*ratio_y[n])+bb2[n])

ind=0
for i in range(len(out_images)):
    if(i in indexes):
        continue
    print(ind)
    cv2.circle(out_images[i],(int(nl1[ind][0]),int(nl6[ind][5])),2,(0,255,0),thickness=1)
    cv2.circle(out_images[i],(int(nl2[ind][1]),int(nl7[ind][6])),2,(0,255,0),thickness=1)
    cv2.circle(out_images[i],(int(nl3[ind][2]),int(nl8[ind][7])),2,(0,255,0),thickness=1)
    cv2.circle(out_images[i],(int(nl4[ind][3]),int(nl9[ind][8])),2,(0,255,0),thickness=1)
    cv2.circle(out_images[i],(int(nl5[ind][4]),int(nl10[ind][9])),2,(0,255,0),thickness=1)
    plt.imshow(out_images[i])
    ind=ind+1
    img.imsave('/home/vyom/ADA_Project/OurModel_V3_torch/OutputImages/OutputImage'+str(i),out_images[i])
with open("/home/vyom/ADA_Project/OurModel_V3_torch/classifiedAs.txt", "a") as myfile:
    myfile.write("-----------------------image --------------------------\n")
    myfile.write("\nGender\n")
    myfile.write(str(genderList[0]))
    myfile.write("\nSmile\n")
    myfile.write(str(smileList[0]))
    myfile.write("\nGlasses\n")
    myfile.write(str(glassesList[0]))
    myfile.write("\nGender\n")
    myfile.write(str(poseList[0]))
    myfile.write("------------------------------------------")