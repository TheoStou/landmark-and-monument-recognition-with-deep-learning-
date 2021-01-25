# Landmark and Monument Recognition with Deep Learning
(Train a custom model based on the monuments of the UNESCO Monuments Route of Thessaloniki with Tensorflow Object Detection API and deploy it into Android.)

In this project, we provided an approach to constructing and implementing a Monument/Landmark object detection model based on Convolution Neural Networks. Gathering a sufficient number of images for the [UNESCO World Heritage Monuments of Thessaloniki](https://thessaloniki.travel/en/exploring-the-city/themed-routes/unesco-monuments-route) and using the SSD MobileNet V1 and V2 models from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), we were able to successfully train and apply them in an Android application, capable of conducting a real-time object detection. </br>


## Table of Contents 


## Approach


### 1. Preparing the dataset
Probably the most time-consuming part of the experiment consisted of the data gathering and the pre-processing. An essential and inextricable task for every training model in machine learning is the collection of the required dataset (collection of data).


#### A. Data Gathering
In order to train the model 4708 images were gathered regarding the 18 different UNESCO World Heritage Monuments, located in the city of Thessaloniki. The data collection was created by manually downloaded images from the web, in combination with personal photographs taken for the respective monuments/landmarks. As lighting conditions and different angles of a monument are two factors that directly affect the coloring and possibly the shape of a monument, we try to maintain a variety in our images. With this strategy, it is ensured a greater chance for a correct and accurate prediction, even under various circumstances. 


#### B. Data pre-processing


- ##### Checking the quality
it is vital to maintain only the images that satisfy a certain resolution. Images with a very low resolution are not useful for our purpose, as the information gained from them is minor and consequently the overall performance is decreasing.


- ##### Renaming the images
In order to avoid any errors regarding the name of the images, which often poses a great length and contain punctuation marks and spaces, we renamed each image providing a meaningful to us name.


- ##### Resizing the images
The resize of the images constitutes a crucial action, especially in case we do not want to overwhelm the model. Furthermore, the reduced resolution will improve dramatically the pre-processing time. Both the SSD_MobileNet_V1_coco and the SSD_MobileNet_V2_coco demand a 300x300 pixels input resolution for the whole amount of images. The following figure demonstrates the method utilized to resize the images located in a specified path, keeping at the same time the original aspect ratio. </br>
```ruby
from PIL import Image
import os, sys

path = "C:/data/images/"
dirs = os.listdir(path)
final_size = 300;

def resize_an_image_keeping_aspect_ratio():
    
    for item in dirs:
         if item == '.DS_Store':
             continue
         if os.path.isfile(path+item):
             im = Image.open(path+item)
             f, e = os.path.splitext(path+item)
             size = im.size
             ratio = float(final_size) / max(size)
             new_image_size = tuple([int(x*ratio) for x in size])
             im = im.resize(new_image_size, Image.ANTIALIAS)
             new_im = Image.new("RGB", (final_size, final_size))
             new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
             new_im.save(f + '.jpg', 'JPEG', quality=90)
```

- ##### Annotating the images
A time-consuming but fundamental part of the pre-processing is the annotation of the images. To annotate the images, we use a tool named [LabelImg](https://tzutalin.github.io/labelImg/), available for several platforms, enabling us an easy draw of the desired bounding box along with the annotation for each box, as seen in the figure provided down below. LabelImg also offers us the opportunity for saving the annotation with either YOLO or PascalVOC format. After the confirmation for the bounding box and the annotation, the program automatically generates a (XML) file with the same name as the image. For each image, we can observe that the (XML) file, besides the filename and the size of the image also contains the label name in association with the coordinates of the drawn bounding box/boxes. </br> </br>
![Example of the Labelimg environment  Creating a bounding box and provide an annotation for it](https://user-images.githubusercontent.com/74372152/105701383-e589de80-5f12-11eb-8766-a8e1acd854fe.png) </br>
Example of the Labelimg environment. Creating a bounding box and provide an annotation for it. </br>


##### Splitting the dataset into train and test set
Before proceeding with the next step, we separate our images and their matching annotation files into a training and a testing set, using an 80-20 ratio. The separation is implemented randomly to help ensure homogeneity, allocating the data into two distinct folders, one created for the train and the other for the test set.
```ruby
import os
from random import choice
import shutil

#arrays to store file names
imgs =[]
xmls =[]

#setup dir names
trainPath = 'C:/Users/Jarvis/Desktop/Dataset/18. Rotunda/train'
testPath = 'C:/Users/Jarvis/Desktop/Dataset/18. Rotunda/test'
crsPath = 'C:/Users/Jarvis/Desktop/Dataset/18. Rotunda/' #dir where images and annotations stored

#setup ratio
train_ratio = 0.8
test_ratio = 0.2

#total count of imgs
totalImgCount = len(os.listdir(crsPath))/2

#soring files to corresponding arrays
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:
        if filename.endswith('.xml'):
            xmls.append(filename)
        else:
            imgs.append(filename)


#counting range for cycles
countForTrain = int(len(imgs)*train_ratio)
countForTest = int(len(imgs)*test_ratio)

#cycle for train dir
for x in range(countForTrain):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.xml' # get name of corresponding annotation file

    #move both files into train dir
    shutil.move(os.path.join(crsPath, fileJpg), os.path.join(trainPath, fileJpg))
    shutil.move(os.path.join(crsPath, fileXml), os.path.join(trainPath, fileXml))

    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)



#cycle for test dir   
for x in range(countForTest):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.xml' # get name of corresponding annotation file

    #move both files into train dir
    shutil.move(os.path.join(crsPath, fileJpg), os.path.join(testPath, fileJpg))
    shutil.move(os.path.join(crsPath, fileXml), os.path.join(testPath, fileXml))

    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)

#summary information after splitting
print('Total images: ', totalImgCount)
print('Images in train dir:', len(os.listdir(trainPath))/2)
print('Images in test dir:', len(os.listdir(testPath))/2)
```


### 2. Google Colaboratory
[Google’s Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) python environment was selected to perform the experiment and [Google Drive](https://www.google.com/intl/en_jm/drive/) to host the necessary data and appropriate tools. </br>
In short, Google Colaboratory [76] or commonly referred to as Google Colab is a research project, offering the potential for students, data scientists, or AI researchers to utilize powerful hardware just like GPUs and TPUs to implement and run their machine learning problems. In addition, Google Colab is based on an interactive Jupyter Notebook framework, equipped with various features [77]. Another main advantage of utilizing its environment is that it has the most frequent libraries for machine learning pre-installed not to mention the fact that also allows us to upload our files and mount our Google Drive.


### 4.3	Setting up the environment
To generate an accurate machine learning model, able to identify the provided monuments/landmarks we used the Tensorflow python library. Although at this time a newer version of Tensorflow has been released, Tensorflow 1.15 was chosen in our case, as it constitutes a more stable and robust solution regarding the newer one which presents some bugs and is still under development. On the official page of [Tensorflow on Github](https://github.com/tensorflow/models/tree/master/research), we can find all the necessary files to train a model. In order to access these files, we opted to clone the Tensorflow Github’s repository to our Google Drive account. To achieve this operation, we first mounted a Google Drive account to Google Colab with the following command:
```ruby
from google.colab import drive
drive.mount('/content/gdrive')
```
and by allowing access inserting the required authorization code. </br> </br>
![Copy-pasting the provided code, will allow access to our Google Drive repository](https://user-images.githubusercontent.com/74372152/105702930-2daa0080-5f15-11eb-981e-d216dcdc58b7.png) </br>
Copy-pasting the provided code, will allow access to our Google Drive repository. </br>


