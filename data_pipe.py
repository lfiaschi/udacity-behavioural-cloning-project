import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn


DOWSAMPLE_FACTOR = .95
SIDE_CORRECTION = .2


def load_meta(test_size = .2):
    """
    Load some metadata for the training and validation samples
    :return: 
    """
    samples = []
    for folder in sorted(glob.glob('data/*_data'),reverse=True):
        folder = os.path.abspath(folder)

        print('Loading data from {}'.format(folder))
        samples_tmp = pd.read_csv('{}/driving_log.csv'.format(folder),encoding='utf8')
        for col in ['left','right','center']:
            samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: os.path.join(folder,x))

        samples.extend(samples_tmp.to_dict('records'))

    print('Total samples : {}'.format(len(samples)))

    if test_size:
        train_samples, validation_samples = train_test_split(samples, test_size=test_size)

    else:
        train_samples, validation_samples = samples, []


    #For the training samples augment the data:
    tmp = list()
    print('Train samples : {}'.format(len(train_samples)))
    print('Validation samples  : {}'.format(len(validation_samples)))


    return train_samples, validation_samples


def generator(samples, batch_size=128 ,
              load_images= True,
              loop_forever=True,
              shuffle = True,
              flip_images=False,
              include_side_images=True,
              downsample_angles_at_zero=False,
              min_angle = -100000,
              max_angle = +100000):

    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = sklearn.utils.shuffle(samples)

        images = []
        angles = []


        for ii in range(0, num_samples):

            batch_sample = samples[ii]
        # for offset in range(0, num_samples, batch_size):
        #     batch_samples = samples[offset:offset+batch_size]
        #
        #     images = []
        #     angles = []
        #
        #     for batch_sample in batch_samples:

            # Remove angles at zero which would weight too much otherwise
            if downsample_angles_at_zero and batch_sample['steering'] == 0 and np.random.rand() <= DOWSAMPLE_FACTOR:
                continue

            # May remove angles which are two wide
            if batch_sample['steering'] < min_angle or batch_sample['steering'] > max_angle:
                continue

            # May add a flipped version of the initial image instead of original image with equal probability
            if not flip_images:
                flip_image_flag = False
            else:
                flip_image_flag = np.random.rand() < .5

            center_angle = float(batch_sample['steering'])
            if flip_image_flag:
                center_angle = -center_angle
            angles.append(center_angle)

            if load_images:
                center_image = cv2.imread(batch_sample['center'])
                if flip_image_flag:
                    center_image = np.fliplr(center_image)
                images.append(center_image)

            # May include the side images
            if include_side_images:
                lr_flag = np.random.rand() > .5
                if load_images:

                    side_img = batch_sample['left'] if lr_flag else batch_sample['right']
                    side_img = cv2.imread(side_img)
                    images.append(side_img)

                angles.append(center_angle + SIDE_CORRECTION if lr_flag else center_angle - SIDE_CORRECTION)

            if len(angles) == batch_size:
                X_train = np.array(images)
                y_train = np.array(angles).squeeze()

                if load_images:
                    assert y_train.ndim == 1, y_train.shape
                    assert X_train.ndim == 4, X_train.shape

                    assert X_train.shape[0] <= batch_size, X_train.shape
                    assert X_train.shape[-1] == 3, 'Last dimension must be channel !!'

                if shuffle:
                    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

                yield X_train, y_train

                images = []
                angles = []


        if not loop_forever: break


def get_generator_length(samples,
                         batch_size=128 ,
                      load_images= True,
                      flip_images=False,
                      include_side_images=True,
                      downsample_angles_at_zero=False,
                      min_angle = -100000,
                      max_angle = +100000
                         ):


    train_generator = generator(samples, batch_size,
                                load_images=load_images,
                                loop_forever=False,
                                shuffle=False,
                                flip_images=flip_images,
                                include_side_images=include_side_images,
                                downsample_angles_at_zero=downsample_angles_at_zero,
                                min_angle=-100000,
                                max_angle=+100000
                                )

    all_angles_modified = list()
    for _, y in train_generator:
        all_angles_modified.append(y)
    all_angles_modified = np.concatenate(all_angles_modified)

    return len(all_angles_modified)


def load_lap(lapname):

    samples_tmp = pd.read_csv('data/{}/driving_log.csv'.format(lapname), encoding='utf8')

    images = list()
    angles = list()
    filenames = list()
    for _,row in samples_tmp.iterrows():
        row['center'] = 'data/{}/{}'.format(lapname, row['center'])
        filenames.append(row['center'])
        center_image = cv2.imread(row['center'])
        images.append(center_image)
        center_angle = float(row['steering'])
        angles.append(center_angle)

    X_train = np.array(images)
    y_train = np.array(angles).squeeze()

    return X_train, filenames, y_train


if __name__ == '__main__':

        # Put files in better format

        for folder in sorted(glob.glob('data/*_data'),reverse=True):
            try:
                folder = os.path.abspath(folder)
                filename = '{}/dr' \
                           'iving_log.csv'.format(folder)
                print('Loading data from {}'.format(folder))
                samples_tmp = pd.read_csv(filename,encoding='utf8')
                samples_tmp.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake',
                                       'speed']

                for col in ['left','right','center']:
                    samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: 'IMG/' + x.split('/')[-1])

                samples_tmp.to_csv(filename,index=False,encoding='utf8')

            except FileNotFoundError:
                pass