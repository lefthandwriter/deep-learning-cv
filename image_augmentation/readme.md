# Image Augmentation using imgaug
Assuming existing images with bounding boxes and labels in PASCAL VOC format (XML files) (ImageNet format), this example generates image augmentations using [imgaug](https://github.com/tzutalin/labelImg). Additionally, new XML annotation files are created for the new images.

Makes use of some code from (labelImg) [https://github.com/tzutalin/labelImg], under labelImg_libs folder.

# Setup
Install imgaug first:
`pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
`git clone https://github.com/aleju/imgaug`
`cd imgaug`
`python setup.py install`

Install lxml:
`pip install lxml`

Other dependencies:
opencv
numpy
pandas

# Usage
A. To generate image augmentations:
1. Create a folder `folder-name` and place all the images and XML annotation files in there.
2. `python augmenter.py folder-name/*jpeg false`

Note: examples for applying single operations (brighter, darker) and multiple operations (blur, contrast normalization and changing brightness) are provided. For the full list of possible operations, see https://github.com/aleju/imgaug.

B. To verify correctness of the annotation of the output (assumed bounding box):
`python verifyBox.py <path-to-augmented-image.jpeg>`

C. To generate a .csv list of train and test images that can be used with the TensorFlow object detection API to generate TFRecords:
`python xml_to_csv.py <path-to-image-folder> <image-extension>`