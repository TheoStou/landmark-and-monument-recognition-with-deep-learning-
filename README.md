# Landmark and Monument Recognition with Deep Learning
(Train a custom model based on the monuments of the UNESCO Monuments Route of Thessaloniki with Tensorflow Object Detection API and deploy it into Android.)

In this project, we provided an approach to constructing and implementing a Monument/Landmark object detection model based on Convolution Neural Networks. Gathering a sufficient number of images for the [UNESCO World Heritage Monuments of Thessaloniki](https://thessaloniki.travel/en/exploring-the-city/themed-routes/unesco-monuments-route) and using the SSD MobileNet V1 and V2 models from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), we were able to successfully train and apply them in an Android application, capable of conducting a real-time object detection. </br>


## Table of Contents 
* [Approach](#approach)
    * [1. Preparing the dataset](#1-preparing-the-dataset)
        * [A. Data Gathering](#a-data-gathering)
        * [B. Data pre-processing](#B-data-pre-processing)
    * [2. Google Colaboratory](#2-google-colaboratory)
    * [3. Setting up the environment](#3-setting-up-the-environment)
    * [4. Converting xml files to tfrecord](#4-converting-xml-files-to-tfrecord)
    * [5. Creating a label map](#5-creating-a-label-map)
    * [6. Download a pre-trained model and alter the configuration file](#6-download-a-pre-trained-model-and-alter-the-configuration-file)
    * [7. Training the model](#7-training-the-model)
    * [8. Measuring and evaluating the model with Tensorboard](#8-measuring-and-evaluating-the-model-with-Tensorboard)
    * [9. Exporting the graph from the trained model](#9-exporting-the-graph-from-the-trained-model)
    * [10. Getting the model's lite version with TFLite](#10-getting-the-model-s-lite-version-with-tFLite)
    * [11. Adding a metadata file to the transformed model](#11-adding-a-metadata-file-to-the-transformed-model)
    * [12. Deploying the model to Android](#12-deploying-the-model-to-android)
* [Results](#results)
* [Applying Data Augmentation](#applying-data-augmentation)
* [Future Work](#future-work)
* [Conclusions](#conclusions)


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


### 3. Setting up the environment
To generate an accurate machine learning model, able to identify the provided monuments/landmarks we used the Tensorflow python library. Although at this time a newer version of Tensorflow has been released, Tensorflow 1.15 was chosen in our case, as it constitutes a more stable and robust solution regarding the newer one which presents some bugs and is still under development. On the official page of [Tensorflow on Github](https://github.com/tensorflow/models/tree/master/research), we can find all the necessary files to train a model. In order to access these files, we opted to clone the Tensorflow Github’s repository to our Google Drive account. To achieve this operation, we first mounted a Google Drive account to Google Colab with the following command:
```ruby
from google.colab import drive
drive.mount('/content/gdrive')
```
and by allowing access inserting the required authorization code. </br> </br>
![Copy-pasting the provided code, will allow access to our Google Drive repository](https://user-images.githubusercontent.com/74372152/105702930-2daa0080-5f15-11eb-981e-d216dcdc58b7.png) </br>
Copy-pasting the provided code, will allow access to our Google Drive repository. </br> </br>
Now that we have granted access to our account, with the following piece of code:
```ruby
import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models
```
we can clone the specified repository. By changing the working directory, we can start installing the appropriate dependencies that are essential for the experiment. </br>
For the implementation of the project, we also constructed the proper working directory. More specifically, we created:
- A “deploy” folder, which contains the pre-trained model, the “labelmap.pbtxt” and the “pipeline_file.config”, which are being covered in the next subsections.
- An “images” folder, which includes the images and the annotations (.xml) files for both the train and the evaluation phase. 
- A “data” folder, required to save some additional files regarding the training process.
- An empty “training” folder, to store the checkpoint files of the training procedure.
- An “exported_model” folder, to save the exported model.


### 4. Converting xml files to tfrecord
Another necessary operation is the conversion of the aforementioned (XML) files, which contain the total amount of the annotations, into (TFRECORD) files. Tfrecord is a type of data supported by Tensorflow and is highly recommended, as it offers an effective way of sustaining a scalable architecture and a common input format. The described process occurred in two sequential steps. We first transformed the (XML) files into (CSV) files following by the transformation of the (CSV) files into (TFRECORD) files. </br> </br>
**Convert XML to CSV.**
```ruby
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
```
**Convert CSV to TFRECORD.**
```ruby
def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
```


### 5. Creating a label map
As a trained model can only identify an integer value, it is essential to create a label map file. Alternatively stated, we have to map each of our class labels into an integer number. Since our project includes 18 monuments/landmarks, the process should be applied to all 18 classes. Starting with the integer “1”, a sample of the generated label map is provided in the following command and is saved with a (PBTXT) format. 
```ruby
item {
  name: "Aghios Demetrius"
  id: 1
}
item {
  name: "Saint Apostoles"
  id: 2
}
item {
  name: "Saint Sophia"
  id: 3
}
item {
  name: "Acheiropoietos"
  id: 4
}
item {
  name: "Saint Aikaterini"
  id: 5
...
```


### 6. Download a pre-trained model and alter the configuration file
Considering that training a complete convolutional neural network from scratch requires significant amounts of time, storing capacity, and computational power, using a pre-trained model is strongly suggested. The idea behind the concept is transfer learning, as referred to at 3.8 and it is being actualized with the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). From the [Model Zoo repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), we can observe various pre-trained models along with their speed and mAP performance pre-trained and tested on the [COCO dataset](https://cocodataset.org/#home). To conduct our experiment, the “SSD_MobilNet_V1_coco” and the “SSD_MobileNet_V2_coco” models were selected, in view of the fact that both present pleasant performance and great speed, particularly important for mobile devices. </br> </br>
The last step before running our custom object detection model is to download and import the appropriate configuration file for this pre-trained model. As the name suggests, the configuration file defines the exact way of the training process and allows us to modify the desired parameters. The required changes we applied are summarized down below: 
- The “num_classes” are defined as equal to 18.
- A “batch_size” of 24 is used.
- The “fine_tune_checkpoint” was set each time regarding the model’s checkpoints name.
- The “input_path” and the “label_map_path” for both “train_input_reader” and “eval_input_reader” are linked with the “train.record”, the “test.record” and the “label_map.pbtxt” files.


### 7. Training the model
It is highly recommended, before start training the model to alter the runtime type to GPU, as Google Colab offering the opportunity to utilize its GPU, which without any doubt can provide significantly faster results. For the training process, the “model_main.py” script from the “object_detection” folder is used, along with several necessary arguments as demonstrated in the following block of code.
```ruby
!python /content/gdrive/My\ Drive/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}
```
More specifically, the following prerequisites are provided:
- The path of the “pipeline_file”, which is identical to the configuration file created at the previous step.
- The “model_dir”, which is the path to the output directory, is needed to store the final checkpoint files.  
- The “num_steps”, which entails the total number of iterations that the experiment will be conducted.
- The “num_eval_steps”, which defines the number of steps before applying an evaluation. 


### 8. Measuring and evaluating the model with Tensorboard
This is an optional action and it does not affect the remaining procedure, however, usually, it is practical and essential to evaluate the model’s performance, so as to have a clearer view. Tensorboard is a tool designed to provide various measurements and visualizations, regarding the trained model, just like the loss and the accuracy values. </br> 
**Loading Tensorboard by providing the training directory of the model.**
```ruby
%load_ext tensorboard
%tensorboard --logdir /content/gdrive/My\ Drive/models/research/object_detection/training
```
Regularly, these metrics are based on the “metric_set” chosen in the configuration file and each selection may offer us a different kind of evaluation. </br> </br>
![Example of the Tensorboard interface](https://user-images.githubusercontent.com/74372152/105705839-6ba92380-5f19-11eb-88c5-27e7258a8a20.png) </br>
Example of the Tensorboard interface.


### 9. Exporting the graph from the trained model
In this step, the “export_tflite_ssd_graph.py” script is used to generate a frozen graph from the created checkpoint files. It constitutes a fundamental procedure as transforms the produced files from the training section into a usable format for testing and deploying our model in Android. A common approach concerning the selection of the appropriate step checkpoint is to take the latest one, as it has covered the most steps. Nonetheless, this may not be the case for any occasion and for this reason, it is recommended to first consult an evaluation metric. If the process of exporting the graph is properly conducted, then two new files will be created at the specified directory, a **“tflite_graph.pb”** and a **“tflite_graph.pbtxt”**.
```ruby
!python /content/gdrive/My\ Drive/models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path /content/gdrive/My\ Drive/models/research/deploy3/pipeline_file.config \
  --trained_checkpoint_prefix /content/gdrive/My\ Drive/models/research/object_detection/training3/model.ckpt-193043 \
  --output_directory /content/gdrive/My\ Drive/models/research/object_detection/exported_model3 \
  --add_postprocessing_op True
```


### 10. Getting the model's lite version with TFLite
In order to deploy our model in the Android platform, it is necessary first to convert the created “tflite_graph.pb” file into a Tensorflow Lite format. Using a special format for manipulating models, Tensorflow Lite offers efficient execution for mobile and other embedded devices, utilizing limited computational power. To perform the conversion, the [Tensorflow Lite converter tool](https://www.tensorflow.org/lite/guide/get_started) was used. There are two options for conversion, either to generate a Non-quantized model or a quantized one. The former of the two presents slightly better results and requires more storing space, although the latter of the two provides slightly worst results requiring fewer storing space. After the correct execution, a (TFLITE) file is created. </br> </br>
**Converting the model to Tensorflow Lite format, using the non-quantized method.**
```ruby
import tensorflow as tf

graph_def_file = "object_detection/exported_model3/tflite_graph.pb"
input_arrays = ["normalized_input_image_tensor"]
output_arrays = [
        'TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
        'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3' ]

converter = tf.lite.TFLiteConverter.from_frozen_graph( 
    graph_def_file, 
    input_arrays, 
    output_arrays, 
    input_shapes={'normalized_input_image_tensor':[1, 300, 300, 3]} )

converter.allow_custom_ops = True
tflite_model = converter.convert()
open("object_detection/exported_model3/detect.tflite", "wb").write(tflite_model)
```


### 11. Adding a metadata file to the transformed model
The final step before the implementation of the model into Android is the creation of a metadata file. Alternatively, a file that provides knowledge about the model which is comprised of both human and machine-readable segments. This file is divided into three subcategories, model, input, and output information. After executing the appropriate commands, the metadata file is attached to the defined model. Optionally, with the MetadataDisplayer tool, we have the potential to visualize the results, writing the metadata file into a (JSON) file. 


### 12. Deploying the model to Android
In this section, [Android Studio 4.1](https://developer.android.com/studio?authuser=1) is used to create and build an application based on our already trained custom model and regarding the aforementioned monuments/landmarks. Briefly, Android Studio grants an integrated development environment for the Android platform, especially constructed for Android development. In order to deploy our custom model, the following procedure was applied: </br>
- Tensorflow’s Lite object detection sample repository was cloned into our local repository.
- Initializing the Android Studio application, “Open an existing Android Studio project” was selected. Subsequently, navigating through our repository, the cloned Android directory is chosen. 
- If requested, we confirm to apply Gradle Sync and we proceed with the installation of various platforms and tools.
- At this point, we search for the asset folder, located in the “Android/app/src/main” directory and we need to change these files with our personal files. These are consist of the “ssd_mobilenet_v2.tflite” which is our trained model and the “labelmap.txt” which contains the labels of our trained classes.
- Several alterations should also be applied at the “DetectorActivity.java” file, so as to deploy our custom model. We change the “TF_OD_API_MODEL_FILE” into our model’s name. The same procedure is followed for the “TF_OD_API_LABELS_FILE” variable, providing the name of our label map file. Additionally, we modify the “TF_OD_API_IS_QUANTIZED” value to “false”, considering that our model is a Non-quantized one. (Figure 35) provides an illustrative example of the “DetectorActivity.java” after the above implementations.
- At the “build.gradle” file, we also select the appropriate “complileSdVersion” and the desired “targetSdkVersion” and we deactivate the auto-download of the model.
- Finally, by connecting our Android device to our personal computer, with the developer mode enabled and by selecting “Make a project” our application is ready for use. Alternatively, the Android Studio environment allows us to create a virtual device offering the opportunity for various tests, possibly at different Android versions. </br> </br>
The following figures represent an example of the produced application, as well as the user’s environment and the first predictions conducted toward the monuments/landmarks. </br> </br>
![Example of the created application, “TFL Detect”](https://user-images.githubusercontent.com/74372152/105738629-e84ef880-5f3f-11eb-8e6d-c7b7888fc07e.png) </br>
Example of the created application, “TFL Detect”. </br> </br>
![Example of the user’s application environment  The picture displays a correct identification of the Church of Saint Sophia in the area of Thessaloniki, along with a bounding box and a confidence score](https://user-images.githubusercontent.com/74372152/105738753-0b79a800-5f40-11eb-8277-f63aef361e0d.png) </br>
Example of the user’s application environment. The picture displays a correct identification of the Church of Saint Sophia in the area of Thessaloniki, along with a bounding box and a confidence score. </br> </br>


## Results
**mAP performance for SSD MobileNet V1**
| **Step** | **mAP** | **mAP_0.50_IoU** | **AP_0.75_IoU** |
| :--- | :--- | :--- | :--- |
| **1622** | 1.189E-02% | 9.358E-02% | 3.67E-06% |
| **24872** | 53.88% | 85.68%| 59.53% |
| **68146** | 64.51% | 92.31% | 76.48% |
| **89725** | 67.36% | 93.35% | 78.30% |
| **126067** | 70.29% | 94.64% | 81.83% |
| **163119** | 67.87% |93.32% | 77.60% |
| **199986** | **71.90%** | **95.05%** | **82.94%** |

**mAP performance for SSD MobileNet V2**
| **Step** | **mAP** | **mAP_0.50_IoU** | **AP_0.75_IoU** |
| :--- | :--- | :--- | :--- |
| **1712** | 3.950E-02% | 3.178E-01% | 1.04E-04% |
| **31621** | 60.73% | 88.73% | 69.71% |
| **65954** | 68.63% | 93.96% | 80.84%|
| **102733** | 72.00% | 95.63% | 83.48% |
| **137659** | 72.70% | **96.20%** | 83.73% |
| **165487** | **73.59%** | 95.73% | 84.50% |
| **197470** | 73.32% | 95.03% | **85.03%** |

</br> </br>
![The mAP ( 50 IOU) of the ssd_mobilenet_v1 model](https://user-images.githubusercontent.com/74372152/105741312-d0c53f00-5f42-11eb-90f9-f9a1515da745.png) </br>
mAP (.50 IoU) of the SSD MobileNet V1 </br> </br>

![The mAP ( 50 IOU) of the ssd_mobilenet_v1 model](https://user-images.githubusercontent.com/74372152/105741421-f05c6780-5f42-11eb-8942-31174a71ba49.png) </br>
mAP (.50 IoU) of the SSD MobileNet V2 </br> </br>

In order to acquire a more rounded overview of the model’s performance, each of these two graphs was tested independently on 844 images from all the 18 monuments and assessed in terms of TP, FP, FN, Precision, Recall, F1-Score, and Accuracy. The aforementioned metrics are presented for both nodels down below. </br> </br>

![Table_1](https://user-images.githubusercontent.com/74372152/105742846-ac6a6200-5f44-11eb-8926-61008252994b.png) </br>
Table of the evaluation results for SSD MobileNet V1 model. </br> </br>

![Table_2](https://user-images.githubusercontent.com/74372152/105744280-8c876e00-5f45-11eb-8a44-b43955023fb4.png) </br>
Table of the evaluation results for SSD MobileNet V2 model. </br> </br>


## Applying Data Augmentation
A common method utilized especially in deep learning to enhance the results for specific tasks is that of Data Augmentation. This approach enables the production of new artificially training data based on the already existed training data. Various data augmentation techniques can be implemented for images just like position augmentation (scaling, cropping, padding, etc.) and color augmentation (brightness, saturation, etc.) offering a greater diversity to the dataset. </br> </br>

As object detection is the investigated subject in our case study, data augmentation should also be applied for the generated bounding boxed of the original dataset. To achieve this task, initially, all the generated (XML) files are converted into (CSV). Sequentially, the augmentation parameters are structured selecting randomly one augmenter for each instance. Furthermore, the transformation of the bounding boxes into a (data frame) format constitutes an essential part, before eventually completing the process by generating the new transformed images along with the new (CSV) file that contains the total amount of the labels (original images + augmented images). </br> </br>

For the purpose of this experiment scale, translate, multiplication, Gaussian Blur, Gaussian Noise, and Linear Contrast were utilized to generate the new training data. (Figure 48) demonstrates an example of the data augmentation that was applied to produce two additional images from an existing image. In addition, a total of 7550 new images were produced to supply the training process, which was conducted based on the SSD MobileNet V2 model, which was the one that presented overall the best outcomes. </br> </br>

NOTE: The code that utilized to implement data augmentation is provided in the **Data_Augmentation.py** file. </br> </br>

![Example of the data augmentation](https://user-images.githubusercontent.com/74372152/105869545-0419d380-6000-11eb-964b-d864099b9a94.png) </br>
Example of the data augmentation technique applied. The church of Panayia Chalkeon is depicted on the three above cases. (a) It is the original image. (b) It is a newly produced image with Linear Contrast. (c) It is another newly produced image with Gaussian Blur. </br> </br>

**mAP performance for SSD MobileNet V2 with augmented data**
| **Step** | **mAP** | **mAP_0.50_IoU** | **AP_0.75_IoU** |
| :--- | :--- | :--- | :--- |
| **1416** | 1.374E-01% | 5.376E-01% | 1.77E-03% |
| **33378** | 61.49% | 91.23% | 69.48% |
| **65672** | 65.59% | 93.65% | 75.10% |
| **103870** | 69.37% | 94.93% | 79.60% |
| **140507** | 71.68% | 96.14% | 83.31% |
| **182708** | 71.14% | 95.56% | 82.41% |
| **196640** | **73.19%** | **96.37%** | **83.37%** |

![Table_3](https://user-images.githubusercontent.com/74372152/105870697-2f50f280-6001-11eb-9155-5cfe6c4e4d47.png) </br>
Table of the evaluation results for SSD MobileNet V2 with data augmentation technique. </br> </br>


## Future Work
To further enrich the effort presented in this project, collecting additional images under various circumstances and from diverse angles of view, particularly for those with lower metrics, would definitely benefit the performance of the models. Combining this action with the incorporation of the GPS feature into the application, it would reduce the erroneous detections to the minimum and upgrade the overall outcome. Of course, offering the potential for learning crucial information about each monument with a build-in function is an action that can only be beneficial.        


## Conclusions
In this project, we have provided an approach to constructing and implementing a Monument/Landmark object detection model based on Convolution Neural Networks. Gathering a sufficient number of images for the UNESCO World Heritage Monuments of Thessaloniki and using the SSD MobileNet V1 and V2 models, we were able to successfully train and apply them in an Android application, capable of conducting a real-time object detection. Utilizing the Transfer Learning technique, both models are able to identify and localize the desired objects with satisfactory accuracy, achieving at the same time high mAP performance. The application operates properly performing predictions consisted of a bounding box that encloses the detected object in combination with a score that describes the confidence of the model. Considering that the models provide accurate results in a short time of period, it strengthens the fact that real-time object detection can be also applied on mobile devices, even with limited computational power. The results presented in chapter 5 confirm the high performance for the investigating models, proving synchronously a superiority of the SSD MobileNet V2 over the SSD MobileNet V1. Moreover, as reasonably proven different monument exhibit different results, with some of them being easier and some other more challenging to identify. The data augmentation technique implemented on the V2 model managed to present a slight increase in the overall accuracy of the model.
















