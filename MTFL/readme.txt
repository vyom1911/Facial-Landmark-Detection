Multi-Task Facial Landmark (MTFL) dataset v1.0
2014.10.28
------------------------------------------------------

Description:

This dataset contains 12,995 face images which are annotated with (1) five facial landmarks, (2) attributes of gender, smiling, wearing glasses, and head pose. The images are from (1) CUHK Face Alignment Database (http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) (2) Annotated Facial Landmarks in the Wild (AFLW) database (http://lrs.icg.tugraz.at/research/aflw/).

------------------------------------------------------

File Description
Annotation.txt : Bounding box annotation of training dataset
Annotation_test.txt : Bounding box annotation of testing dataset
validation_bb.txt : Should contain bounding box detail of images you want to annotate with facial landmarks
training.txt: contains annotation of landmarks for training dataset
testing.txt: contains annotation of landmarks for testing dataset
validate.txt: contains annotation of landmarks for user given images - not required

-------------------------------------------------------
Format:
The dataset is divided into two parts: training and testing. The annotations are stored in according text files (training.txt and testing.txt). Each line in the text file is for one face image. The format is:
#image path #x1...x5,y1..y5 #gender #smile #wearing glasses #head pose

--x1...x5,y1...y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
--gender: 1 for male, 2 for female
--smile: 1 for smiling, 2 for not smiling
--glasses: 1 for wearing glasses, 2 for not wearing glasses.
--head pose: 1 for left profile, 2 for left, 3 for frontal, 4 for right, 5 for right profile

------------------------------------------------------

Citations:
[1] Zhanpeng Zhang, Ping Luo, Chen Change Loy, Xiaoou Tang. Facial Landmark Detection by Deep Multi-task Learning, in Proceedings of European Conference on Computer Vision (ECCV), 2014

[2] Zhanpeng Zhang, Ping Luo, Chen Change Loy, Xiaoou Tang. Learning and Transferring Multi-task Deep Representation for Face Alignment. Technical report, arXiv:1408.3967, 2014.
