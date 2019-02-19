import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import h5py
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

dataPath = 'RealImageNet/ImageNetSubsample/Data/CLS-LOC'
trainPath = os.path.join(dataPath, 'train')

outputModelPath = os.path.join('output','saved-models')
outputFigPath = os.path.join('output', 'figs')
h5Path = os.path.join('output', 'image-h5')
bboxesPath = os.path.join('output', 'bboxes.csv')
lines = [line.rstrip('\n').split() for line in open('RealImageNet/LOC_synset_mapping.txt')]
wnids_to_words = {line[0]:' '.join(line[1:]) for line in lines }


img_width, img_height = 224, 224

"""
Creates any necessary output dirs that we will need. 'figs', 'saved-models', 'image-h5'
"""
def make_output_dirs(dirs):
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)



"""
Determines which files have bounding boxes associated with them, 
and outputs a .csv with: class_id, filename, box.

Can read with pd.read_csv('output/bboxes.csv')
Can get arrays into array format with:
x = df.boxes[0]
import ast
ast.literal_eval(x)

Can get files for a specific folder (class id with):

df[df.ids == 'n02017213'].files.unique()
"""

def parseBoundingBoxes():
    ids = []
    files = []
    boxlist = []
    
    boxesPath = os.path.join("RealImageNet", "LOC_train_solution.csv")
    imageBoxes = [line.rstrip('\n').split(',') for line in open(boxesPath)][1:]

    for boxes in imageBoxes:
        imageFileName = boxes[0] #gets the filename of the image with a bounding box or boxes
                
        boxesStringSplit = boxes[1].split() #splits the string of boxes into a space-separated array

        for i in range(0, len(boxesStringSplit), 5):
            
            box = boxesStringSplit[i:i+5] #gets each bounding box in form: [class_id xmax ymax xmin ymin]
            ids.append(box[0]) 
            files.append(imageFileName+".JPEG")
            boxlist.append(box[1:])
        
    df = pd.DataFrame({'ids': ids, 'files': files, 'boxes': boxlist})
    df.to_csv(bboxesPath)


"""
Reads image data from the training folder, applies augmentation,
and saves all image arrays to .h5 files. Each .h5 has datasets: 'x_train', 'y_train, 'x_val', 'y_val'.

NOTE: ONLY PARSES IMAGES THAT HAVE BOUNDING BOXES

'_train': contains 80% of all original data
'_val': contains 20% of all original data


"""
def parseImages(folders, saveToH5 = True, img_width=224, img_height=224):
    if not os.path.exists(bboxesPath):
        print("Parsing bounding boxes before image data....")
        parseBoundingBoxes()
        print("...done")
    
    boxesDF = pd.read_csv(bboxesPath)
    for folder in folders:
        print('Parsing images for class ID: {0}'.format(folder))
        x_train = []
        y_train = []
        imageFiles = boxesDF[boxesDF.ids == folder].files.unique()
        for imageFile in imageFiles:
            imagePath = os.path.join(trainPath, folder, imageFile)
            if os.path.exists(imagePath):
                img = image.load_img(imagePath, target_size=(img_width, img_height))
                img = image.img_to_array(img)/255.
                x_train.append(img)
                y_train.append(folder)
               
        df_train = pd.DataFrame({'x': x_train, 'y': y_train})
        df_train, df_val = train_test_split(df_train, test_size=0.2)
         
        if saveToH5:
            path = os.path.join(h5Path, folder + '.h5')
            output = h5py.File(path, 'w') 
            output.create_dataset('x_train', data = list(df_train.x))
            output.create_dataset('y_train', data = df_train.y.astype('S'))
            output.create_dataset('x_val', data = list(df_val.x))
            output.create_dataset('y_val', data = df_val.y.astype('S'))
            output.close()
        

"""
Loads a single h5 file, from the default h5 output path, given a training class id.

The file should already exist and have been created with parseImages, or it will return empty. 
"""

def loadH5(id):
    x_t, y_t, x_v, y_v = [], [], [], []
    path = os.path.join(h5Path, id + ".h5")
    if os.path.exists(path):
        data = h5py.File(path, 'r')
        x_t, y_t, x_v, y_v =  data['x_train'][:], data['y_train'][:], data['x_val'][:], data['y_val'][:]
        data.close()
    else:
        print("NO SUCH FILE {0} EXISTS. Run parseImages([{1}]) to fix the issue.".format(path, id))
    
    y_t = [y.decode('utf-8') for y in y_t]
    y_v = [y.decode('utf-8') for y in y_v]
    return x_t, y_t, x_v, y_v 

"""
Loads a list of h5 files, from the default h5 output path, given a list of training class id.

The file should already exist and have been created with parseImages, or it will be skipped over. 
"""
def loadH5s(ids):
    x_train, y_train, x_val, y_val = [],[],[],[]

    for id in ids:
        x_t, y_t, x_v, y_v = loadH5(id)
        x_train.extend(x_t)
        y_train.extend(y_t)
        x_val.extend(x_v)
        y_val.extend(y_v)

    return x_train, y_train, x_val, y_val 

def translateID(id):
    return wnids_to_words[id]

def getClassLabels():
    boxesDF = pd.read_csv(bboxesPath)
    labels = boxesDF.ids.unique()
    return labels


"""
Augments data and returns x_aug, y_aug
"""
def augmentData(x, y, augments = []):
    x_aug, y_aug = [], []
    for i in range(0, len(x)):
        img = x[i]
        for aug in augments:
            augmented = aug.augment_image(img)
            x_aug.append(augmented)
            y_aug.append(y[i])
    
    return x_aug, y_aug

def displayImage(x):
    plt.imshow(x)


def topClasses(prediction, classes, n=3):
    idx = np.argsort(prediction)[-n:][::-1]
    tups = []
    for i in idx:
        tups.append((translateID(classes[i]), prediction[i]))
    return tups


def init():
    print("Checking to make sure output directories are created..")
    make_output_dirs([outputModelPath, outputFigPath, h5Path])
    print("..done")