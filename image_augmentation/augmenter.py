import imgaug as ia
from imgaug import augmenters as iaa

import cv2
import numpy as np

import random, string
import os, glob, ntpath, sys
import xml.etree.ElementTree as ET

from labelImg_libs.pascal_voc_io import PascalVocWriter, PascalVocReader

ia.seed(1)

class Augment:
	def __init__(self):
		self.img_orig_filename = None # to store current original image filename
		self.xml_orig_filename = None # to store current original xml filename

	def load_xml_and_img(self, filename):
		"""
		Load the image into memory and parse its corresponding xml file for
		bounding box.

		Parameters:  filename:   provide the path to the image
		Returns   :  img     :   image array
					 bbox    :   bounding box ia object (xmin,ymin,xmax,ymax)
		"""
		img_name = filename
		xml_name = os.path.splitext(img_name)[0] + ".xml"
		self.img_orig_filename = filename
		self.xml_orig_filename = xml_name

		img = cv2.imread(img_name)
		xroot = ET.parse(xml_name).getroot()

		xbox = xroot.find('object').find('bndbox')
		bbox = []
		for child in xbox:
			bbox.append(child.text)
		bbox = ia.BoundingBoxesOnImage([
			ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
			], shape=img.shape)

		return img, bbox

	def create_augmentation_multi(self, img, bbox):
		"""
		Apply in random order, a sequntial order of these augmentation
		Parameters:  img     :   image array
					 bbox    :   bounding box ia object (xmin,ymin,xmax,ymax)
		Returns   :  img_aug :   augmented image array
					 bbx_aug :   augmented bounding box ia object (xmin,ymin,xmax,ymax)
		"""
		seq = iaa.Sequential([
			iaa.AverageBlur(k=(2, 11)), # blur the image
			iaa.ContrastNormalization((0.5,1.5)), # changes contrast
			iaa.Multiply((0.5, 1.5)), # changes brightness
			], random_order=True)
		seq_det = seq.to_deterministic()
		img_aug = seq_det.augment_images([img])[0]
		bbs_aug = seq_det.augment_bounding_boxes([bbox])[0]

		return img_aug, bbs_aug

	def create_augmentation_brighter(self, img, bbox):
		"""
		Apply a particular augmentation
		Parameters:  img     :   image array
					 bbox    :   bounding box ia object (xmin,ymin,xmax,ymax)
		Returns   :  img_aug :   augmented image array
					 bbx_aug :   augmented bounding box ia object (xmin,ymin,xmax,ymax)
		"""
		seq_det = iaa.OneOf([
				iaa.Multiply((1.2, 2.0)), # brighter
			])
		img_aug = seq_det.augment_images([img])[0]
		bbs_aug = seq_det.augment_bounding_boxes([bbox])[0]

		return img_aug, bbs_aug

	def create_augmentation_affine(self, img, bbox):
		"""
		TODO: If the affine transformation causes the object to be beyond the image boundary,
		change the label to None.

		Apply a particular augmentation
		Parameters:  img     :   image array
					 bbox    :   bounding box ia object (xmin,ymin,xmax,ymax)
		Returns   :  img_aug :   augmented image array
					 bbx_aug :   augmented bounding box ia object (xmin,ymin,xmax,ymax)
		"""
		seq_det = iaa.OneOf([
				iaa.Affine(scale=(0.5, 2.0)), # affine transformation (scaling)
			])
		img_aug = seq_det.augment_images([img])[0]
		bbs_aug = seq_det.augment_bounding_boxes([bbox])[0]

		return img_aug, bbs_aug

	def create_augmentation_darker(self, img, bbox):
		"""
		Apply a particular augmentation
		Parameters:  img     :   image array
					 bbox    :   bounding box ia object (xmin,ymin,xmax,ymax)
		Returns   :  img_aug :   augmented image array
					 bbx_aug :   augmented bounding box ia object (xmin,ymin,xmax,ymax)
		"""
		seq_det = iaa.OneOf([
				iaa.Multiply((0.2, 0.9)), # darker
			])
		img_aug = seq_det.augment_images([img])[0]
		bbs_aug = seq_det.augment_bounding_boxes([bbox])[0]

		return img_aug, bbs_aug

	def save_xml_and_img(self, img_aug, bbs_aug, fmt):
		"""
		Saves the augmented image to file in the provided format, and writes the
		augmented bounding boxes to an xml file with corresponding name
		Parameters: img_aug  :   image array
					bbs_aug  :   bounding box ia object (xmin,ymin,xmax,ymax)
					fmt      :	 string type, image extension format
									(allowed list: "jpeg", "png", "jpg")
		"""
		if fmt!="jpeg" and fmt!="png" and fmt!="jpg":
			raise ValueError("format should be either jpeg, jpg or png")
		else:
			save_img_name = os.path.splitext(self.img_orig_filename)[0]
			head, tail = ntpath.split(save_img_name)
			rnd_string = ''.join(random.choices(
									string.ascii_uppercase + string.digits,
									k=10)) # create random string for unique name
			save_img_name = tail + "_" + rnd_string + os.extsep + fmt # join the format
			sub_dir = os.path.join(head, "augmented") # create a subdir under the data folder
			print(sub_dir)
			if not os.path.exists(os.path.dirname(sub_dir)):
				os.makedirs(os.path.dirname(sub_dir))
				print("made sub_dir")
			save_img_name = os.path.join(head, "augmented", save_img_name)
			save_xml_name = os.path.splitext(save_img_name)[0] + ".xml"
			print(save_img_name)
			print(save_xml_name)

			# Write image to file
			cv2.imwrite(save_img_name, img_aug)

			# Write xml to file
			imgFolderPath = sub_dir
			imgFolderName = os.path.split(imgFolderPath)[-1]
			imgFileName = os.path.basename(save_img_name)

			imageShape = [img_aug.shape[0], img_aug.shape[1],
				3 if img_aug.shape[2] is 3 else 1]
			writer = PascalVocWriter(imgFolderName, imgFileName,
								imageShape, localImgPath=save_img_name)

			# Read the existing xml file, change the bounding box, write the rest to file
			reader = PascalVocReader(self.xml_orig_filename)
			reader.parseXML()

			for box in bbs_aug.bounding_boxes:
				writer.addBndBox(int(box.x1),
								int(box.y1),
								int(box.x2),
								int(box.y2),
								reader.label,
								0)
			writer.save(targetFile=save_xml_name)

	def run(self, foldername, debug=False):
		"""
		Run pipeline.
		Parameters: foldername: Path to folder containing images
		"""
		file_list = glob.glob(foldername)
		if debug:
			file_list = file_list[0:4] ## [DEBUG]

		for file in file_list:
			img, bbox = self.load_xml_and_img(file)

			## Run multiple operations at once
			img_aug, bbs_aug = self.create_augmentation_multi(img, bbox)
			self.save_xml_and_img(img_aug, bbs_aug, "jpeg")

			## Run individual augmentations
			img_aug, bbs_aug = self.create_augmentation_brighter(img, bbox)
			self.save_xml_and_img(img_aug, bbs_aug, "jpeg")

			img_aug, bbs_aug = self.create_augmentation_darker(img, bbox)
			self.save_xml_and_img(img_aug, bbs_aug, "jpeg")

			## Don't use affine yet - bounding box not accurate
			# img_aug, bbs_aug = self.create_augmentation_affine(img, bbox)
			# self.save_xml_and_img(img_aug, bbs_aug, "jpeg")

def main(foldername, debugMode):
	aug = Augment()
	aug.run(foldername, debugMode)

if __name__ == '__main__':
	if len(sys.argv) > 2:
		main(sys.argv[1], sys.argv[2])
	else:
		raise ValueError("usage: python augmenter.py <path-to-folder-with-image-extension> <debugMode>. \
			Example: python augmenter.py data/original/*jpeg false")

