from models import LeNetKerasMSE, Simple, NvidiaNet
from data_pipe import *
from keras.callbacks import EarlyStopping
from math import ceil
from matplotlib import pyplot as plt

plt.switch_backend('agg')

outmodelname = 'models/model.h5'

ch, row, col = 3, 160, 320  # Image shape
batch_size = 256
keep_prob = .9
min_angle = -1000.0
max_angle = 1000.0

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

train_samples, validation_samples = load_meta_new(exclude_track=None)

train_generator = generator_new(train_samples,
                                batch_size,
                                keep_prob=keep_prob,
                                min_angle = min_angle,
                                max_angle = max_angle)

validation_generator = generator_validation(validation_samples,
                                            batch_size=batch_size)

model = NvidiaNet((row,col,ch), dropout=.3)

model.compile(loss='mse', optimizer='adam')

steps_per_epoch = len(train_samples) // batch_size
validation_steps = len(validation_samples) // batch_size

print('Number of training steps {}'.format(steps_per_epoch))
print('Number of validation steps {}'.format(validation_steps))

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps,
                                     epochs=100,
                                     callbacks=[early_stopping],
                                     verbose = 1
                                     )

print('Saving model to {}'.format(outmodelname))
model.save(outmodelname)

# Saving the learning curve
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('models/learning_curve.png')