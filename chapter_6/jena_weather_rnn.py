

########################################
# Read data
########################################
import os

data_dir = '../data/jena_climate'

fname = os.path.join(data_dir, "jena_climate_2009_2016.csv")

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

########################################
# Convert the data to a usable form
########################################

import numpy as np

float_data = np.zeros((len(lines), len(header) -1))

# make a list_comp of the rows in the data
# unpack them into each row of float_data
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values


########################################
# Plot the temperature time series
########################################

from matplotlib import pyplot as plt

# get column 1 and all it's rows
temp = float_data[:, 1]

plt.plot(range(len(temp)), temp)
plt.show()

#data was collected every ten minutes, so this is 10 days
plt.plot(range(1440), temp[:1440])
plt.show()

########################################
# Normalize the data points
########################################

# we only want to do a mean from the first
# 200000 timesteps since this will be training set
training_timestep = 200000

mean = float_data[:training_timestep].mean(axis=0)
float_data -= mean
std = float_data[:training_timestep].std(axis=0)
float_data /= std

########################################
# make the data generator function
########################################

def generator(data, lookback, delay, min_index, max_index, shuffle=False,
              batch_size=128, step=6):
              
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)

        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indicies = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indicies]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


########################################
# make test, training, and validation generator
########################################

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=training_timestep,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# // divide without remainder
val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size

########################################
# Non-Machine learning Baseline
########################################

def evaluate_naive_methods():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    #
    print("value of mae", np.mean(batch_maes))
    print("Average absolute error", np.mean(batch_maes) * std[1])

evaluate_naive_methods()

########################################
# Machine learning baseline 
########################################

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)


########################################
# Plot loss by epoch
########################################

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

#go from 1 to len loss + 1 because range only goes to end -1
epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label="Validation")
plt.title('Training and validation loss Flatten/LSTM')
plt.legend()

plt.show()

print(model.metrics_names)
#print("Print the loss and accuracy", model.evaluate(input_test, y_test))

########################################
# Replace LSTM With GRU layer remove flatten
########################################

model = Sequential()
#we remove the flatten because so we retain the timeseries data
#the GRU layer is less computationally expensive but less representationally
# powerful then LSTM
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

########################################
# Plot to evaluate model with GRU
########################################

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss GRU')
plt.legend()

plt.show()

########################################
# Model with dropout and recurrent dropout in the 
# GRU layer
########################################

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)


########################################
# Plot with dropout
########################################

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss GRU')
plt.legend()

plt.show()



