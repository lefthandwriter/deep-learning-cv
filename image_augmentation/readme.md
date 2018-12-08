# Image Augmentation
Esther Ling, 2018

# Overview
Assuming existing images with bounding boxes and labels in PASCAL VOC format (XML files), this example:

- generates image augmentations using [imgaug](https://github.com/tzutalin/labelImg)
- automatically generates correspnding XML annotation files for the new images
- has a helper script to verify correctness of the position of the generated bounding box
- generates a .csv list of train and test images under a 80%-20% random split, which can be used with the TensorFlow object detection API


# Setup
Install imgaug first:

`pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`

`git clone https://github.com/aleju/imgaug`

`cd imgaug`

`python setup.py install`

Other dependencies:

- lxml
- opencv
- numpy
- pandas

# Usage
A. To generate image augmentations:
1. Create a folder `folder-name` and place all the images and XML annotation files in there.
2. `python augmenter.py folder-name/*jpeg false`

Note: examples for applying single operations (brighter, darker) and multiple operations (blur, contrast normalization and changing brightness) are provided. For the full list of possible operations, see https://github.com/aleju/imgaug.

B. To verify correctness of the annotation of the output (assumed bounding box):

`python verifyBox.py <path-to-augmented-image.jpeg>`

C. To generate a .csv list of train and test images that can be used with the TensorFlow object detection API to generate TFRecords:

`python xml_to_csv.py <path-to-image-folder> <image-extension>`


# Notes
- imgaug seems to have trouble with computing the correct bounding box location under affine transformations


# Related
1. I made use of some code from [labelImg](https://github.com/tzutalin/labelImg), under labelImg_libs folder.
2. TensorFlow object detection API: https://github.com/tensorflow/models/tree/master/research/object_detection.