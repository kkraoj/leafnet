import cv2
import numpy as np
import os
import pandas as pd
import scipy.misc
import utils
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
DATA_SOURCE = 'leafsnap'
NUM_CLASSES = 185
RESOLUTION = 224
ROT_ANGLE = 90 #cnagle by which to rotate augmented images in training set



columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data_file = '{}-dataset-images.csv'.format(DATA_SOURCE)
data = pd.read_csv(data_file, names=columns, header=1)

test_df = data.loc[data.source=='field'].sample(frac=0.5, random_state=7)
## fraction increased because lab images are only 7k
train_df = data.drop(test_df.index)


images_train_original = train_df['image_pat'].tolist()
images_train_segmented = train_df['segmented_path'].tolist()
images_train = {'original': images_train_original, 'segmented': images_train_segmented}
species_train = train_df['species'].tolist()
species_classes_train = sorted(set(species_train))

images_test_original = test_df['image_pat'].tolist()
images_test_segmented = test_df['segmented_path'].tolist()
images_test = {'original': images_test_original, 'segmented': images_test_segmented}
species_test = test_df['species'].tolist()
species_classes_test = sorted(set(species_test))

print('\n[INFO]  Training Samples : {:5d}'.format(len(images_train['original'])))
print('\tTesting Samples  : {:5d}'.format(len(images_test['original'])))
print('[INFO] Processing Images')

os.chdir('dataset_all/{}'.format(DATA_SOURCE))
def save_images(images, species,  directory='train', \
                csv_name='temp.csv', augment=False):
    cropped_images = []
    image_species = []
    image_paths = []
    count = 1
    write_dir = 'dataset/{}'.format(directory)
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    for index in range(len(images['original'])):
        image = utils.load_image_and_preprocess(
            images['original'][index], images['segmented'][index], resolution = RESOLUTION)
        if type(image) != type([]):
            image_dir = '{}/{}'.format(write_dir, species[index].lower().replace(' ', '_'))
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

            file_name = '{}.jpg'.format(count)

            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_dir, file_name), image_to_write)
            image_paths.append(os.path.join(image_dir, file_name))
            cropped_images.append(image)
            image_species.append(species[index])
            count += 1

            if augment:
                angle = ROT_ANGLE
                while angle < 360:
                    rotated_image = utils.rotate(image, angle)

                    file_name = '{}.jpg'.format(count)
                    image_to_write = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
    #                    cv2.imwrite(os.path.join(image_dir, file_name), image_to_write)
                    result = Image.fromarray((image_to_write).astype(np.uint8))
                    result.save(os.path.join(image_dir, file_name))
                    image_paths.append(os.path.join(image_dir, file_name))
                    cropped_images.append(rotated_image)
                    image_species.append(species[index])

                    angle += ROT_ANGLE
                    count += 1

        if index > 0 and index % 1000 == 0:
            print('[INFO] Processed {:5d} images'.format(index))

    print('[INFO] Final Number of {} Samples: {}'.format(directory, len(image_paths)))
    raw_data = {'image_paths': image_paths,
                'species': image_species}
    df = pd.DataFrame(raw_data, columns = ['image_paths', 'species'])
    df.to_csv(csv_name)
#os.chdir(r'D:\Krishna\DL\leafnet')
#save_images(images_train, species_train, directory='train',
#            csv_name='leafsnap-dataset-train-images.csv', augment=True)
save_images(images_test, species_test, directory='test',
            csv_name='leafsnap-dataset-test-images.csv', augment=False)

print('\n[DONE]')
