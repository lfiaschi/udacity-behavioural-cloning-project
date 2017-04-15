import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

def load_meta(test_size = .2):
    """
    Load some metadata for the training and validation samples
    :return: 
    """
    samples = []
    for folder in glob.glob('data/*_data'):
        folder = os.path.abspath(folder)

        print('Loading data from {}'.format(folder))
        samples_tmp = pd.read_csv('{}/driving_log.csv'.format(folder),encoding='utf8')
        for col in ['left','right','center']:
            samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: os.path.join(folder,x))

        samples.extend(samples_tmp.to_dict('records'))

    print('Total samples : {}'.format(len(samples)))

    train_samples, validation_samples = train_test_split(samples, test_size=test_size)

    print('Train samples : {}'.format(len(train_samples)))
    print('Validation samples  : {}'.format(len(validation_samples)))

    return train_samples, validation_samples


def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample['center'])
                center_angle = float(batch_sample['steering'])

                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles).squeeze()

            assert y_train.ndim == 1, y_train.shape
            assert X_train.ndim == 4, X_train.shape

            assert X_train.shape[0] <= batch_size, X_train.shape
            assert X_train.shape[-1] == 3, 'Last dimension must be channel !!'

            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    # Little test

    train_samples,_ = load_meta()
    train_generator = generator(train_samples, batch_size=32)

    for x,y in train_generator:
        print(x.shape)
        print(y.shape)
        break

    from matplotlib import pyplot as plt
    plt.imshow(x[0])
    plt.show()