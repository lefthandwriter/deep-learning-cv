"""
Verify the position of the bounding box after augmentation

Usage:
	python verifyBox.py <path-to-augmented-image.jpeg>
"""
import cv2
import os, sys
import xml.etree.ElementTree as ET
import numpy as np

if len(sys.argv) < 2:
	raise ValueError("usage: python verifyBox.py <path-to-augmented-image.jpeg>. \
			Example: python verifyBox.py data/images/hd_v1_image-076_QJMB6OOVMG.jpeg")
else:
	img_name = sys.argv[1]
	xml_name = os.path.splitext(img_name)[0] + ".xml"
	print(xml_name)

	# Read image
	img = cv2.imread(img_name)
	# Read xml file
	xroot = ET.parse(xml_name).getroot()

	xbox = xroot.find('object').find('bndbox')
	bbox = []
	for child in xbox:
		bbox.append(child.text)
	bbox = np.asarray(bbox, dtype='int32')

	cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
	cv2.imshow("augmented_image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()