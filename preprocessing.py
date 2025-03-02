import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.preprocessing import image
from PIL import Image

import h5py
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
import utils

import pandas_ml as pml

dataPath = 'RealImageNet/ImageNetSubsample/Data/CLS-LOC'
trainPath = os.path.join(dataPath, 'train')
baseModelName = "model-{0}-{1}-{2}"
trainhistory = os.path.join('output', 'trainhistory')
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
                with Image.open(imagePath) as x:
                    width, height = x.size
                    if width >= img_width and height >= img_height:
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
                hf['x_train'].resize((hf['x_train'].shape[0] + df_train.x.shape[0], img_width, img_height, 3))
                
                hf['x_train'][-df_train.x.shape[0]:] = list(df_train.x)

                hf['y_train'].resize((hf['y_train'].shape[0] + df_train.y.shape[0]), axis=0)
                hf['y_train'][-df_train.y.shape[0]:] = df_train.y.astype('S')

                hf['x_val'].resize((hf['x_val'].shape[0] + df_val.x.shape[0], img_width, img_height, 3))
                hf['x_val'][-df_val.x.shape[0]:] = list(df_val.x)

                hf['y_val'].resize((hf['y_val'].shape[0] + df_val.y.shape[0]), axis=0)
                hf['y_val'][-df_val.y.shape[0]:] = df_val.y.astype('S')
    
    print("Wrote {0} images to {1}".format(imgCount, path))



            


def loadH5(filename):
    """
Loads a single h5 file, from the default h5 output path. 

The file should already exist and have been created with parseImages, or it will return empty. 
"""
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


def loadH5s(ids):
    """
Loads a list of h5 files, from the default h5 output path, given a list of training class id.

The file should already exist and have been created with parseImages, or it will be skipped over. 
"""
    x_train, y_train, x_val, y_val = [],[],[],[]

    for id in ids:
        x_t, y_t, x_v, y_v = loadH5(id + ".h5")
        x_train.extend(x_t)
        y_train.extend(y_t)
        x_val.extend(x_v)
        y_val.extend(y_v)

    return x_train, y_train, x_val, y_val 

def translateID(id):
    """
    Simply looks up the raw imagenet id and provides the human-readable label. 
    """
    return wnids_to_words[id]

def getClassLabels():
    """
    Gets all class ids that have bounding boxes. 
    """
    boxesDF = pd.read_csv(bboxesPath)
    labels = boxesDF.ids.unique()
    return labels



def augmentData(x, y, augments = [], p=1.0, replace=False):
     """
    Augments data by shuffling augments and looping through data. Applies 1 random augment to an image with probability p. 
    """
    x_new, y_new = np.copy(x), np.copy(y)
    
    x_aug, y_aug = [], []
    
    for i in range(0, len(x)):   
        r = np.random.uniform()
        if r <= p:
            img = x[i]
            np.random.shuffle(augments)
            for aug in augments:

                augmented = aug.augment_image(img)
                if replace:
                    x_new[i] = augmented
                else:
                    x_aug.append(augmented)
                    y_aug.append(y[i])
        
        
    if not replace:
        x_new = np.concatenate((x_new, x_aug))
        y_new = np.concatenate((y_new, y_aug))
    
    return x_new, y_new

def augmentData2(x, y, augments = [], p=1.0, replace=False):
    """
    Augments data by shuffling augments and looping through data. Applies 1 random augment to an image with probability p. 
    """
    x_old, y_old = np.copy(x), np.copy(y)
    x_aug, y_aug = [], []
    idxs = [i for i in range(0, len(x))]
    np.random.shuffle(idxs)
    idxs = idxs[0:int(p*len(x))]
    for idx in idxs:
        np.shuffle(augments)
        for aug in augments:
            augmented = aug.augment_image(x[idx])
            if not replace:
                x_aug.append(augmented)
                y_aug.append(y[idx])
            else:
                x_old[idx] = augmented
    if not replace:
        x_old = np.concatenate((x_old, x_aug))
        y_old = np.concatenate((y_old, y_aug))
        return x_old, y_old
    
    return x_old, y_old

def displayImage(x):
    """
    Convenience function that displays an image with pyplot. 
    """
    plt.imshow(x)

def displayImageWithBox(x, coords, label):
    """
    Displays an image with a rectangular box placed around a section of the image, given by coords: (xmin, xmax, ymin, ymax)
    """
    fig,ax = plt.subplots(1)
    xmin, xmax, ymin, ymax = coords
    ax.imshow(x)
    rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1.0})
    plt.show()


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
    """
    Simple initialization to make sure output directories are made beforehand.
    """
    print("Checking to make sure output directories are created..", end="", flush=True)
    make_output_dirs([outputModelPath, outputFigPath, h5Path, trainhistory])
    print("..done")




def shuffleH5(filepath, inplace=False):
    """
    Shuffles the training data in an h5, in place. 
    """
    outpath = os.path.join(h5Path, 'shuffle.h5')

    data = h5py.File(filepath, 'r+')


    indexes = np.arange(data['x_train'].shape[0])
    np.random.shuffle(indexes)
    for key in data.keys():
        if key in ['x_train', 'y_train']:
            d = data[key]
            feed = np.take(data[key], indexes, axis=0)
            d[...] = feed
        
    data.close()
    
    
    return filepath

class DataGenerator(keras.utils.Sequence):
    """
    Original data generator. Does not shuffle the whole dataset, instead selects the indices of the H5 file and 
    randomly shuffles them, then accesses a set of indices according to the batch size. LESS EFFICIENT THAN DATAGENERATOR3. 
    """
    def __init__(self, h5file, classes, batch_size=32, isValidation = False, shuffle=False, augmentations = [], augment_pct=0.25):
        self.h5file = h5file
        self.isValidation = isValidation
        
        self.set = 'train'
        if self.isValidation:
            self.set = 'val'
        with h5py.File(self.h5file, 'r') as db:
            self.data_length = len(db['x_'+ self.set])
            
               
        
        self.classes = classes
        self.encoder = LabelBinarizer()
        self.encoder = self.encoder.fit(self.classes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_sets = []
        self.batch_num = 0

        self.augmentations = augmentations
        self.augment_pct = augment_pct

        self.on_epoch_end()

    def __len__(self):
        'Calculates how many steps in an epoch'
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        idxs = list(self.index_sets[self.batch_num])
        with h5py.File(self.h5file, 'r') as db:
            
            x = db['x_'+self.set][idxs]
            y = db['y_'+self.set][idxs]
        
        y = [k.decode('utf-8') for k in y]
        
        
        #TODO
        if len(self.augmentations) > 0:
            x_aug, y_aug = augmentData(x, y, augments = self.augmentations, p=self.augment_pct)
            x.extend(x_aug)
            y.extend(y_aug)

        #TESTING FOR MULTIPROCESSING
        nan_check = np.isnan(x)
        x_new = []
        y_new = []
        for i in range(len(nan_check)):
            if True in nan_check[i]:
                print('NAN @ INDEX {0}'.format(i))
                print('Index num: {0}'.format(idxs[i]))
            else:
                x_new.append(x[i])
                y_new.append(y[i])
        
        x = x_new
        y = y_new

    
        x = np.array(x)
        y = self.encoder.transform(y) 
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



class DataGenerator3(keras.utils.Sequence):
    """
    Generates batches of data. Additional arguments to shuffle the data every epoch and augment the data. 

    Use with keras.model.fit_generator(...) 
    """

    def __init__(self, h5file, classes, batch_size=32, isValidation = False, 
        shuffle=False, augmentations = [], augmentReplace = True, augment_pct=0.25):
        self.baseh5file = h5file
        self.h5file = h5file
        self.isValidation = isValidation
        self.set = 'train'
        if self.isValidation:
            self.set = 'val'
        

        with h5py.File(self.h5file, 'r') as db:
            self.data_length = len(db['x_'+self.set])
            
        
        self.classes = classes
        self.encoder = LabelBinarizer()
        self.encoder = self.encoder.fit(self.classes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_sets = []
        self.batch_num = 0

        self.augmentations = augmentations
        self.augmentReplace = augmentReplace
        self.augment_pct = augment_pct
        self.on_epoch_end()

    def __len__(self):
        'Calculates how many steps in an epoch'
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'


        with h5py.File(self.h5file, 'r') as db:
            
            start = self.batch_num * self.batch_size
            end = min(self.data_length, start+ self.batch_size)

            x = db['x_'+self.set][start:end]
            y = db['y_'+self.set][start:end]
            
        
        y = [k.decode('utf-8') for k in y]
        x = list(x)
        
        #TODO
        if len(self.augmentations) > 0:
            if not self.augmentReplace:
                x_aug, y_aug = augmentData(x, y, augments = self.augmentations, p = self.augment_pct)
                x.extend(x_aug)
                y.extend(y_aug)
            else:
                x, y = augmentData(x, y, augments = self.augmentations, p = self.augment_pct, replace = True)

        #TESTING FOR MULTIPROCESSING
        nan_check = np.isnan(x)
        x_new = []
        y_new = []
        for i in range(len(nan_check)):
            if True in nan_check[i]:
                print('NAN @ INDEX {0}'.format(i))
            else:
                x_new.append(x[i])
                y_new.append(y[i])
        
        x = x_new
        y = y_new

    
        x = np.array(x)
        y = self.encoder.transform(y) 
        y = np.array(y)
        

        self.batch_num +=1

        return x,y
        

    def on_epoch_end(self):
        
        if self.set == 'train' and self.batch_num==0 and self.shuffle:
            self.h5file = shuffleH5(self.baseh5file, inplace=True)
        
        self.batch_num = 0



def predictionsToDataframe(model, x_val, y_val, encoder):

    """
    Calculates model predictions and places them in a dataframe. Shows top1-5 predicitons against the truth label. 
    """
    y_val = [k.decode('utf-8') for k in y_val]
    y_val = encoder.transform(y_val)
    y_val = np.array(y_val)
    x_val = np.array(x_val)

    predictions = model.predict(x_val)
    one = []
    two = []
    three = []
    four = []
    five = []
    for p in predictions:
        top = topClasses(p, encoder.classes_, n=5)
        one.append(top[0][0])
        two.append(top[1][0])
        three.append(top[2][0])
        four.append(top[3][0])
        five.append(top[4][0])

    df = pd.DataFrame({'truth': [translateID(x) for x in encoder.inverse_transform(y_val)],
                      'one': one,
                      'two': two,
                      'three': three,
                      'four': four,
                      'five': five}) 

    return df



def perClassAccuracy(model, x_val, y_val, encoder):

    """
    Returns a dataframe with all classes sorted by top-1 accuracy, but displays all top-1 to top-5 accuracies, per class.
    """
    df = predictionsToDataframe(model, x_val,  y_val, encoder)
    labels = df.truth.unique()
    accs = [accuracies(df[df.truth == label]) for label in labels]

    acc_df = pd.DataFrame({'label': labels, 'accuracies': accs})
    acc_df[['top-1', 'top-2', 'top-3', 'top-4', 'top-5']] = pd.DataFrame(acc_df['accuracies'].values.tolist())
    acc_df = acc_df.drop('accuracies', 1)
    acc_df = acc_df.sort_values(by=['top-1'])

    return acc_df



#USAGE

def top5accuracies(model, x_val, y_val, encoder):
    """
    Computes Top 5 accuracies of a model, given (x_val, y_val) and the encoder that encodes the labels in y_val. 
    h5db = h5py.File(h5file, 'r')
    x_val = np.array(h5db['x_val'])
    top3accuracies(model, h5db['x_val][:], h5db[y_val][:])
    """
    df = predictionsToDataframe(model, x_val, y_val, encoder)
    
    acc1 = len(df[df.truth == df.one])/len(df)


    acc2 = len(df[(df.truth == df.one) | (df.truth == df.two)])/len(df)


    acc3 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) ]) / len(df)

    acc4 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) |  (df.truth == df.four)]) / len(df)

    acc5 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) |  (df.truth == df.four) | (df.truth == df.five)]) / len(df)
    return [acc1, acc2, acc3, acc4, acc5]

def accuracies(df):
    acc1 = len(df[df.truth == df.one])/len(df)


    acc2 = len(df[(df.truth == df.one) | (df.truth == df.two)])/len(df)


    acc3 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) ]) / len(df)

    acc4 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) |  (df.truth == df.four)]) / len(df)

    acc5 = len(df[(df.truth == df.one) | (df.truth == df.two) | (df.truth == df.three) |  (df.truth == df.four) | (df.truth == df.five)]) / len(df)
    
    return [acc1, acc2, acc3, acc4, acc5]

def confusion_matrix(truth, pred):
    conf = pml.ConfusionMatrix(truth, pred)
    return conf

def display_confusion(conf):
    conf.plot()
    plt.show()
    

