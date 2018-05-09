from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from Scripts import baseline_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from sklearn.model_selection import train_test_split

batch_size = 64
original_dim = 340
latent_dim = 2
intermediate_dim = 200
epochs = 200
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
h = Dense(intermediate_dim, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train on float dataset
df = pd.read_csv('Data/train_data.csv')
df2 = df.drop(['IsReadmitted'], axis=1)
df2 = df2.iloc[:, 1:]
baseline_models.center(df2)
x_train, x_test = train_test_split(df2, test_size=0.2, random_state=42)

history = vae.fit(x_train, verbose=2, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
predictions = encoder.predict(df2)
df_new_data = pd.DataFrame(predictions)
df_new_data[df_new_data.shape[1]] = df.IsReadmitted
id_ = df.icustay_id
df_new_data = pd.concat([id_, df_new_data], axis=1)
df_new_data.rename(columns={340: 'IsReadmitted'}, inplace=True)

from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=25,
                               max_depth=25,
                               min_samples_leaf=50)

# train_data = pd.read_csv('Data/train_data.csv')
train_data = df_new_data.iloc[:, 1:]
col = train_data.shape[1]
X_train, y_train = train_data.iloc[:, 0: col - 1], train_data.iloc[:, col - 1]
scores = cross_val_score(clf, X_train, y_train, cv=5)

