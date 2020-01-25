#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

if len(sys.argv) == 1:
#initializing base variables
    val_data_type = "val2017" #type of annotations for validation dataset
    train_data_type = "train2017" ##type of annotations for test dataset
    train_dataset_dir = '/media/dawid/MAJKI/train2017' #directory of training photos diretory
    test_dataset_dir = '/home/dawid/Desktop/human_images' #directory of validation photos diretory
    real_test_dir = '../real_test' #directory of photos for real test
    model_path = '../initial_weights/initial_weights.h5' #initial weights for network

elif len(sys.argv) > 1 and len(sys.argv) != 7:
    sys.exit("Arguments you need to pass: val_data_type, train_data_type, train_dataset_dir, test_dataset_dir, real_test_dir, model_path")

else:
     val_data_type = sys.argv[1]
     train_data_type = sys.argv[2]
     train_dataset_dir = sys.argv[3]
     test_dataset_dir = sys.argv[4]
     real_test_dir = sys.argv[5]
     model_path = sys.argv[6]

# In[2]:


# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '../Mask_RCNN'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib


# In[3]:


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[4]:


class HumanRecognitionConfig(Config):
    """Configuration for training on the human dataset.
    Derives from the base Config class and overrides values specific
    to the human dataset.
    """
    # Give the configuration a recognizable name
    NAME = "human"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (human)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 200

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    BACKBONE = 'resnet50'

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000

config = HumanRecognitionConfig()
config.display()


# In[5]:


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        coco_json = annotation
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"

        class_id = 1
        class_name = 'human'
        if class_id < 1:
            print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
            return

        self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:

            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


# In[6]:


#getting images from coco dataset
from pycocotools.coco import COCO
import numpy as np


def load_annotation_dict(dataDir='..', dataType='val2017'):
    """ Load image information from coco dataset.
    Args:
        dataDir: directory of annotation directory
        dataType: type of annotation file
    Returns:
        annotations_dict: list of annotations
    """
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds );
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    imagesAll = coco.loadImgs(imgIds)
    print("Length of dataset: ",len(imagesAll))

    annotations = []
    for x in imagesAll:
        annIds = coco.getAnnIds(imgIds=x['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        annotations.append(anns[0])

    annotations_dict = {}
    annotations_dict['images'] = imagesAll
    annotations_dict['annotations'] = annotations

    return annotations_dict





# In[7]:

#creating annotation maps basing on previously chosed images
val_annotations_dict = load_annotation_dict(dataType=val_data_type)
train_annotations_dict = load_annotation_dict(dataType=train_data_type)

#preparating train and validation dataset
dataset_train = CocoLikeDataset()
dataset_train.load_data(train_annotations_dict, train_dataset_dir)
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data(val_annotations_dict, test_dataset_dir)
dataset_val.prepare()


# In[8]:

if len(sys.argv) > 1:
    dataset = dataset_train
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[ ]:


# Which weights to start with
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
if len(sys.argv) > 1:
    start_train = time.time()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,
                layers='heads')
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')


# In[ ]:


class InferenceConfig(HumanRecognitionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85

inference_config = InferenceConfig()


# In[ ]:


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)


# In[ ]:


# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = '../initial_weights/initial_weights.h5'

if len(sys.argv) > 1:
    model_path = model.find_last()
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


import skimage

image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], figsize=(5,5))
