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
import matplotlib.pyplot as plt
import sys
import cv2
logging.basicConfig(filename='example.log',level=logging.DEBUG)



#set Keras session to tensorflow to initiate from placeholder
def progress(epoch,loss,accuracy, count, total, status=''):
    total = round(total)
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[Epoch:%s][Training Loss:%s][Testing Error Rate: %s][%s] %s%s ...%s\r' % (epoch,loss,accuracy,bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


class TCDCNN(nn.Module):

    def __init__(self):
         super(TCDCNN, self).__init__()
         self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
         self.conv2 = nn.Conv2d(16,48,kernel_size=3)
         self.conv3 = nn.Conv2d(48,64,kernel_size=3)
         self.conv4=  nn.Conv2d(64,64,kernel_size=2)
         self.linear_1 =nn.Linear(256,10)
         self.linear_2 = nn.Linear(256,2)
         self.linear_3 = nn.Linear(256,2)
         self.linear_4 = nn.Linear(256,2)  
         self.linear_5 = nn.Linear(256,5)
         self.dropout = nn.Dropout();
    
    def loss(self,pred,y):
        criterion = nn.CrossEntropyLoss()
        mse_criterion  = nn.MSELoss();

        #landmark = y[:,0:10].float()
  
        #gender = gender.view(-1,gender.size()[0])

        #smile = y[:,12:14].view(1,2).long()
        #glass = y[:,14:16].view(1,2).long()
        #pose =  y[:,16:21].view(1,5).long()
        loss_mse = mse_criterion(pred[0],y[0])
        loss_gender =criterion(pred[1],y[1])
        loss_smile = criterion(pred[2],y[2])
        loss_glass = criterion(pred[3],y[3])
        loss_pose = criterion(pred[4],y[4])

        loss  = loss_mse - loss_gender - loss_smile - loss_glass - loss_pose
        return loss
    def classifier(self,x):
        x_one = self.linear_1(x)
        x_two = F.softmax(self.linear_2(x))
        x_three = F.softmax(self.linear_3(x))
        x_four = F.softmax(self.linear_4(x))
        x_five = F.softmax(self.linear_5(x))
        return x_one,x_two,x_three,x_four,x_five

    def features(self,x):

        x = F.max_pool2d(F.hardtanh(self.conv1(x)),  2)
        x = F.max_pool2d(F.hardtanh(self.conv2(x)), 2)
        x = F.max_pool2d(F.hardtanh(self.conv3(x)),  2)
        x_tanh = self.conv4(x)
        x = F.hardtanh(x_tanh)
      
        x = x.view( -1,256)
        x =  self.dropout(x)
        return x;
       
    def forward(self,x):
        x  = self.features(x)
        x_one,x_two,x_three,x_four,x_five = self.classifier(x)
        return x_one,x_two,x_three,x_four,x_five

    def predict(self,x):
        x  = self.features(x)
        x_one,x_two,x_three,x_four,x_five = self.classifier(x)
        return [x_one,x_two,x_three,x_four,x_five]

    def accuracy(self,x,y):
        landmarkeye= y[:,[0,5]].float()
        landmarkeyey= y[:,[1,6]].float()
        landmark = y[:,0:10].float()

        mse_criterion  = nn.MSELoss();
        loss_mse = mse_criterion(x,landmark)
        loss_base= mse_criterion(landmarkeye,landmarkeyey)
        accuracytemp= torch.div(loss_mse.data,loss_base.data)
        accuracy= torch.div(accuracytemp, 0.01)
        error=accuracy.numpy()
        error_text = "error rate(%):{}".format(float(error))
        logging.info(error_text)
        
#        print("accuracy: ", accuracy.numpy(),"%")
#        print("base: ", loss_base)
#        logging.info("base")
#        logging.info(float(loss_base.data.numpy()))
        return accuracy[0]

def test_model(model,dataloader_test):
	for i,data in  enumerate(dataloader_test,1):  
		images,landmark,gender,smile,glass,pose = data
		#images = images.squeeze(1)
		landmark = A.Variable(landmark)
		images  = A.Variable(images)
		gender = A.Variable(gender)
		smile = A.Variable(smile)
		glass = A.Variable(glass)
		pose = A.Variable(pose)
		#images = images.squeeze(1)
		#images_temp  = images.data.cpu().numpy()
		#images_temp = images_temp.reshape(40,40)
		x_one,x_two,x_three,x_four,x_five = net(images.float())
		logging.info("Testing commencing for prediction")
		logging.info(x_one)
		logging.info("Testing commencing for actual")
		logging.info(landmark.float());
		logging.info("end-------------------------->")
		#loss = net.loss([x_one,x_two,x_three,x_four,x_five],[landmark.float(),gender.long(),smile.long(),glass.long(),pose.long()])
		accuracy = net.accuracy(x_one,landmark.float())
		return accuracy
    
if __name__ == "__main__":

    net = TCDCNN();
    optim = optim.SGD(net.parameters(),0.003)
    #input = Variable(torch.randn(1, 1, 32, 32))
    
    count = 0;
    accuracy = 0;
    loss = 0;
    mypath = "/home/vyom/ADA_Project/MTFL"
    train= "training.txt"
    annotation = "annotation.txt"
    test = "testing1.txt"
    annotation_test = "annotation_test.txt"
    faceLandMark  = FaceLandmarksDataset(mypath,train,annotation)
    dataloader = DataLoader(faceLandMark, batch_size=64,
                            shuffle=True, num_workers=4)
    faceLandMark_test = FaceLandmarksDataset(mypath,test,annotation_test)
    dataloader_test = DataLoader(faceLandMark_test, batch_size=64,
                            shuffle=True, num_workers=4)
    batch_size=64
    
    print("Starting training")
    loss_total = 0;
    
    epochs=100
    
    final_accuracy=0
    final_loss=0
    final_training_accuracy=0
    for e in range(0,epochs):
        total_accuracy = 0
        loss_total=0
        count=0
        total_accuracy_training=0
        for i,data in  enumerate(dataloader,1):  
            images,landmark,gender,smile,glass,pose = data
    		#images = images.squeeze(1)
            landmark = A.Variable(landmark)
            images  = A.Variable(images)
            gender = A.Variable(gender)
            smile = A.Variable(smile)
            glass = A.Variable(glass)
            pose = A.Variable(pose)
            images_temp = images.clone()
            optim.zero_grad()
            x_one,x_two,x_three,x_four,x_five = net(images.float())		
            loss = net.loss([x_one,x_two,x_three,x_four,x_five],[landmark.float(),gender.long(),smile.long(),glass.long(),pose.long()])
            loss.backward();
            optim.step()
    		#loss for training
            loss_total = loss_total +  loss[0].data.numpy()[0]
            accuracy_training = net.accuracy(x_one,landmark.float());
            count = count + 1
            loss_text = "Loss:{}".format(float(loss_total/count))
            logging.info(loss_text)
    		#Accuracy for testing
            net.eval()
            accuracy = test_model(net,dataloader_test);
            total_accuracy = total_accuracy+accuracy
            #Accuracy for training
            total_accuracy_training = total_accuracy_training+accuracy_training
            net.train()
            #progress(str(e),str(loss[0].data.numpy()[0]),str(accuracy),i,int(len(dataloader.dataset)/batch_size))
        print("loss: ", loss_total/count)
        print("error rate: ", total_accuracy/count)
        final_loss = final_loss+loss_total/count
        final_accuracy= final_accuracy+total_accuracy/count
        final_training_accuracy= final_training_accuracy + total_accuracy_training/count
    print("Final loss:",final_loss/epochs)
    print("Final training error rate: ",final_training_accuracy/epochs)
    print("Final testing Error rate: ",final_accuracy/epochs)
    #torch.save(net,'tcdcn1.pt')
