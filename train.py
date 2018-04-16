import os
import sys
import numpy as np
import matplotlib.pyplot as plt


#uncomment following if you want to print full numpy arrays when printing them
#np.set_printoptions(threshold=np.nan)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import img_as_bool
from skimage.transform import resize

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


TRAIN_PATH = '/home/vajira/Desktop/Data mining Bowl challenge/stage1_train/'
train_ids = next(os.walk(TRAIN_PATH))[1]

#manage configurations

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #batch size will depend on this

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100 #1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

    MAX_GT_INSTANCES = 500

    DETECTION_MAX_INSTANCES = 500


config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class myDataset(utils.Dataset):
    def load_mask(self,img_id):
        """Load instance masks for the given image.

                Different datasets use different ways to store masks. Override this
                method to load instance masks and return them in the form of am
                array of binary masks of shape [height, width, instances].

                Returns:
                    masks: A bool array of shape [height, width, instance count] with
                        a binary mask per instance.
                    class_ids: a 1D array of class IDs of the instance masks.
                """




        path = TRAIN_PATH + train_ids[img_id]

        #getting actual image size
        image_path = path + '/images/' + train_ids[img_id] + '.png'
        img_size = (imread(image_path)).shape
            # img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            # X_train[n] = img
        mask_orig = np.zeros((img_size[0], img_size[1], len(next(os.walk(path + '/masks/'))[2])), dtype=np.bool)
        mask1 = np.zeros((img_size[0], img_size[1], 1), dtype=np.bool)
        count = 0
        for mask_file in next(os.walk(path + '/masks/'))[2]:

            img = imread(path + '/masks/' + mask_file)
            mask_ = img.astype(np.bool)
            #     mask = np.maximum(mask, mask_)
            # Y_train[n] = mask
            #mask_n = np.maximum(mask_, mask1)
            #print(mask_orig.shape)
            mask_orig[:, :, count] = mask_
            count += 1
        class_ids = np.array([1 for s in range(len(next(os.walk(path + '/masks/'))[2]))])
        #print(class_ids.shape)
        return mask_orig, class_ids.astype(np.int32)


    def load_image(self, image_id):
        path = TRAIN_PATH + train_ids[image_id]
        img = imread(path + '/images/' + train_ids[image_id] + '.png')[:, :, :3]
        return img

    def load_shapes(self, count, height=None, width=None):
        self.add_class("shapes", 1, "neuclei")

        for i in range(count):
            path = TRAIN_PATH + train_ids[i] + '/images/' + train_ids[i] + '.png'
            img_size = (imread(path)).shape
            self.add_image("shapes", image_id=train_ids[i], path=path, width=img_size[1], height=img_size[0])



#create a training dataset
dataset_t = myDataset()
dataset_t.load_shapes(670) #correct number of images in the training path should be given as input to the function call ex:670
dataset_t.prepare()

#let's create a validation dataset (usually a portion of training dataset will be selected)
dataset_v = myDataset()
dataset_v.load_shapes(620) #
dataset_v.prepare()



#create a model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

#let's train the model
model.train(dataset_t, dataset_v,
            learning_rate=config.LEARNING_RATE,
            epochs=75,
            layers='all')

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually (don't forget to set correct path to save weights)
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights("/home/vajira/PycharmProjects/matterport/mask_rcnn_shapes.h5")


