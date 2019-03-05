import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.preprocessing import image
import h5py
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import random
import time
import utils

dataPath = 'RealImageNet/ImageNetSubsample/Data/CLS-LOC'
trainPath = os.path.join(dataPath, 'train')
baseModelName = "model-{0}-{1}-{2}"

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
def parseImagesOLD(folders, saveToH5 = True, img_width=224, img_height=224):
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
Parse images to one giant h5 file. Should shuffle entries. 
"""
def parseImages(folders, filename, img_width=224, img_height=224):
    if not os.path.exists(bboxesPath):
        print("Parsing bounding boxes before image data....")
        parseBoundingBoxes()
        print("...done")
    imgCount = 0
    path = os.path.join(h5Path, filename)
    boxesDF = pd.read_csv(bboxesPath)
    for folder in folders:
        print('Parsing images for class ID: {0}. '.format(folder), end="", flush=True)
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
        imgCount += len(x_train)
        print("Found {0} images. ".format(len(x_train)))
        df_train = pd.DataFrame({'x': x_train, 'y': y_train})
        df_train, df_val = train_test_split(df_train, test_size=0.2)
         
        
        print("Writing to h5 file: {0}".format(path))
        if not os.path.exists(path):
            output = h5py.File(path, 'w')
            output.create_dataset('x_train', data = list(df_train.x), chunks=True, maxshape=(None,img_width, img_height, 3))
            output.create_dataset('y_train', data = df_train.y.astype('S'),chunks=True, maxshape=(None,))
            output.create_dataset('x_val', data = list(df_val.x),chunks=True, maxshape=(None,img_width, img_height, 3))
            output.create_dataset('y_val', data = df_val.y.astype('S'),chunks=True, maxshape=(None,))
            output.close()
        else:
            with h5py.File(path, 'a') as hf:
                # print(hf['x_train'].shape)
                hf['x_train'].resize((hf['x_train'].shape[0] + df_train.x.shape[0], img_width, img_height, 3))
                # print(len(x_train))
                # print(df_train.x.shape)
                
                hf['x_train'][-df_train.x.shape[0]:] = list(df_train.x)

                hf['y_train'].resize((hf['y_train'].shape[0] + df_train.y.shape[0]), axis=0)
                hf['y_train'][-df_train.y.shape[0]:] = df_train.y.astype('S')

                hf['x_val'].resize((hf['x_val'].shape[0] + df_val.x.shape[0], img_width, img_height, 3))
                hf['x_val'][-df_val.x.shape[0]:] = list(df_val.x)

                hf['y_val'].resize((hf['y_val'].shape[0] + df_val.y.shape[0]), axis=0)
                hf['y_val'][-df_val.y.shape[0]:] = df_val.y.astype('S')
    
    print("Wrote {0} images to {1}".format(imgCount, path))

"""no going to work. need to shuffle all datasets to the same index. """
def shuffleH5(filename):
    path = os.path.join(h5Path, filename)
    datasets = ['x_train', 'y_train', 'x_val', 'y_val']
    with h5py.File(path, 'r+') as h5f:
        for dataset in datasets:
            print('Shuffling dataset: {0}'.format(dataset))
            t1 = time.time()
            random.shuffle(h5f[dataset])
            t2 = time.time()
            print('Time to shuffle {:.3f} seconds'.format(str(t2-t1)))

"""
Loads a single h5 file, from the default h5 output path. 

The file should already exist and have been created with parseImages, or it will return empty. 
"""

def loadH5(filename):
    x_t, y_t, x_v, y_v = [], [], [], []
    path = os.path.join(h5Path, filename)
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
        x_t, y_t, x_v, y_v = loadH5(id + ".h5")
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

"""
Takes in a prediction array (the kind Keras model.predict() produces), and a list of classes.

Returns a list of n tuples, corresponding to the top-n probable classes as judged by predictions. 
Also decodes class labels into their true value. 
"""
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

class DataGenerator(keras.utils.Sequence):

    def __init__(self, h5db, classes, batch_size=32, isValidation = False, shuffle=True, augmentations = []):
        self.h5db = h5db
  
        if isValidation:
            self.X = self.h5db['x_val']
            self.Y = self.h5db['y_val']
        else:
            self.X = self.h5db['x_train']
            self.Y = self.h5db['y_train']
        
        self.data_length = len(self.X)
        
        self.classes = classes
        self.encoder = LabelBinarizer()
        self.encoder = self.encoder.fit(self.classes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_sets = []
        self.batch_num = 0

        self.augmentations = augmentations

        self.on_epoch_end()

    def __len__(self):
        'Calculates how many steps in an epoch'
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        idxs = list(self.index_sets[self.batch_num])
        x = self.X[idxs]
        y = self.Y[idxs]
        y = [k.decode('utf-8') for k in y]
        
        
        #TODO
        if len(self.augmentations) > 0:
            x_aug, y_aug = augmentData(x, y, augments = self.augmentations)
            x.extend(x_aug)
            y.extend(y_aug)

        y = self.encoder.transform(y) 
        
        x = np.array(x)
        y = np.array(y)
        
        self.batch_num +=1

        return x,y
        

    def on_epoch_end(self):
        if self.shuffle:
            idxs = np.random.permutation(self.data_length)
        else:
            idxs = np.arange(0, self.data_length)
        
        sets = utils.chunks(idxs, self.batch_size)
        self.index_sets = [np.sort(s) for s in sets]
        
        self.batch_num = 0