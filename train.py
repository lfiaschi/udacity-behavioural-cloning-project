from models import LeNetKerasMSE, Simple
from data_pipe import load_meta, generator
from keras.callbacks import EarlyStopping
from math import ceil
from matplotlib import pyplot as plt

plt.switch_backend('agg')

outmodelname = 'models/model.h5'

ch, row, col = 3, 160, 320  # Image shape
batch_size = 128

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

train_samples, validation_samples = load_meta()
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(train_samples, batch_size=batch_size)

model = Simple((row,col,ch))

model.compile(loss='mse', optimizer='adam')


steps_per_epoch = ceil(len(train_samples) / batch_size)
validation_steps = ceil(len(validation_samples) / batch_size)

print('Number of training steps {}'.format(steps_per_epoch))
print('Number of validation steps {}'.format(validation_steps))

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_data=validation_generator,
                                     validation_steps=validation_steps,
                                     epochs=5,
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