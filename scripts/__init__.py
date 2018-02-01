import os

# Test if working dir is currnet one and change if this is the case
this_path = os.path.dirname(os.path.realpath(__file__))
working_path = os.getcwd()
if this_path == working_path:
    os.chdir(os.path.dirname(this_path))

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'h5',"mask_rcnn_coco.h5")