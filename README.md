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


### 10. Getting the model’s lite version with TFLite













