from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers, optimizers
import h5py
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import pickle

# pca = PCA(n_components=7)
RANDOM_SEED = 42

df = pd.read_csv("Data/encode_train_data.csv")
df = df.iloc[:, 1:]
df2 = pd.read_csv("Data/encode_test_data.csv")
df2 = df2.iloc[:, 1:]

input_dim = df.shape[1] - 1

X = df.iloc[:, 0:df.shape[1]-1]
y = df.iloc[:, df.shape[1]-1]
X_test = df2.iloc[:, 0:df2.shape[1]-1]
y_test = df2.iloc[:, df2.shape[1]-1]
# X = pca.fit_transform(X)
# X_test = pca.transform(X_test)

lr = 0.00005
batch_size = 1000
epochs = 1000
# input_dim = 7

# create model
model = Sequential()
model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
adam = optimizers.Adam(lr=lr)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# fit
filepath = "nn_zecord.{epoch:02d}-{val_acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
log_dir = '/tmp/tensorflow_logs/nn'
callbacks_list = [checkpoint, TensorBoard(log_dir=log_dir)]
# Fit the model
history = model.fit(X, y, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks_list)

from keras.utils import plot_model
plot_model(model, to_file='dl_simple.png', show_shapes=True, show_layer_names=True)

model.summary()


model.load_weights('nn_weights.178-0.6247.hdf5')
test = pd.read_csv("Data/test_data.csv")
X_test, y_test = test.iloc[:, 1:test.shape[1]-1], test.iloc[:, test.shape[1]-1]
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

model.evaluate(X_test, y_test)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# with open('/trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)


