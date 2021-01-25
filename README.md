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
The resize of the images constitutes a crucial action, especially in case we do not want to overwhelm the model. Furthermore, the reduced resolution will improve dramatically the pre-processing time. Both the SSD_MobileNet_V1_coco and the SSD_MobileNet_V2_coco demand a 300x300 pixels input resolution for the whole amount of images. The following figure demonstrates the method utilized to resize the images located in a specified path, keeping at the same time the original aspect ratio. </br> </br>
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
A time-consuming but fundamental part of the pre-processing is the annotation of the images. To annotate the images, we use a tool named [LabelImg](https://tzutalin.github.io/labelImg/), available for several platforms, enabling us an easy draw of the desired bounding box along with the annotation for each box, as seen in the figure provided down below. LabelImg also offers us the opportunity for saving the annotation with either YOLO or PascalVOC format.


### 2. Google Colaboratory



