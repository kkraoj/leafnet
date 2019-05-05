import numpy as np
import os
import pandas as pd
import scipy.misc
from IPython.display import display
from PIL import Image
from scipy import integrate
from scipy import misc
from scipy import stats
from skimage import img_as_float
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# GLOBAL CONSTANTS
DATA_SOURCE = 'foliage'
RESOLUTION = 224
ROT_ANGLE = 90 #cnagle by which to rotate augmented images in training set

os.chdir('dataset_all/{}'.format(DATA_SOURCE))

print('[INFO] Processing Images')

def save_images(original_directory = 'train_original', save_directory='train', augment=False):
    count = 1
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    for (dirpath, dirnames, filenames) in os.walk(original_directory): 
        dirpath_new = dirpath.replace(original_directory, save_directory)
        if not os.path.exists(dirpath_new):
            os.mkdir(dirpath_new)
        for filename in filenames:
            filepath = dirpath + os.sep + filename
            filepath_new = filepath.replace(original_directory, save_directory).replace(".tif",".jpg")
            image = Image.open(filepath).resize((RESOLUTION, RESOLUTION), Image.ANTIALIAS)
            image.save(filepath_new)
#            result = Image.fromarray((image).astype(np.uint8))
#            result.save(filepath_new)
            count += 1

            if augment:
                angle = ROT_ANGLE
                while angle < 360:
                    rotated_image = image.rotate(angle)

                    filepath_new = filepath.replace(original_directory, save_directory).\
                                    replace('.jpg','_rot_{}.jpg'.format(angle))
                    rotated_image.save(filepath_new)

                    angle += ROT_ANGLE
                    count += 1

                    if count > 0 and count % 100 == 0:
                        print('[INFO] Processed {:5d} images'.format(count))

    print('[INFO] Final Number of {} Samples: {}'.format(save_directory, count))


save_images(original_directory = 'train_rearranged', save_directory='train', augment=True)
save_images(original_directory = 'test_rearranged', save_directory='test', augment=False)
print('\n[DONE]')
