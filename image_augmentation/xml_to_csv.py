"""
This script creates a train-test split (80-20) from the provided list of images
and creates two csv files (train, test) containing image metadata (width, height, class, and bounding box)

Usage:
    python xml_to_csv.py <path-to-image-folder> <image-extension>

Example:
    python xml_to_csv.py data/images/ .jpeg

"""
import os, sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import random

def xml_to_csv(path):
    """
    Accepts a list of the training / test image paths. Parses through each of the coresponding xml files
    and creates a csv file containing the image metadata for each image in the list.
    Parameter:   path: List of training / test image paths
    """
    xml_list = []
    for file in path:
        xml_file = os.path.splitext(file)[0] + ".xml"
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
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

if len(sys.argv) < 3:
    raise ValueError("usage: python xml_to_csv.py <path-to-folder-of-images> <image-format>. \
            Example: python xml_to_csv.py data/images .jpeg")
else:
    foldername = sys.argv[1]
    if sys.argv[2][0] == '.':
        foldername = os.path.join(foldername, ("*" + sys.argv[2]))
    else:
        foldername = os.path.join(foldername, ("*." + sys.argv[2]))
    print("foldername: %s"%foldername)
    file_list = glob.glob(foldername)
    file_list = [os.path.join(os.getcwd(), file) for file in file_list]
    random.shuffle(file_list)
    train_list = file_list[0: int(0.8*len(file_list))]
    test_list  = file_list[int(0.8*len(file_list)):]
    print("Size train set: %d"%len(train_list))
    print("Size test set: %d"%len(test_list))

    if not os.path.exists("data/labels"):
        os.makedirs("data/labels")
        print("made labels directory")

    # Create csv file
    for directory in ["train", "test"]:
        xml_df = xml_to_csv(train_list)
        xml_df.to_csv('data/labels/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')
