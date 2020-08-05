import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, ReLU, Dropout, Activation, Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from paths_safebooru import base_vgg_model, base_vgg_weights, model_path, tags_path, data_path


BATCH_SIZE = 32
EPOCHS = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE
df = pd.read_json(data_path)
labels = df['labels'].to_list()
with open(tags_path, 'r') as infile:
    data = json.load(infile)
    tags = data['tags_list']

if os.path.isfile(base_vgg_model) and os.path.isfile(base_vgg_weights):
    print('Loading VGG19 model')
    with open(base_vgg_model, 'r') as file:
        base_json = file.read()
    base = model_from_json(base_json)
    base.load_weights(base_vgg_weights)
else:
    base = VGG19(include_top=False, weights='imagenet', pooling='max')


def VGG(base_model):
    input1 = Input((256, 256, 3))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input1)
    x = base_model(x)
    x = Dense(2048, name='fc1')(x)
    x = ReLU(name='act1')(x)
    x = Dense(2048, name='fc2')(x)
    x = ReLU(2048)(x)
    x = Dense(len(tags), name='final')(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=input1, outputs=x)
    return model


print('Loading train dataset')
train_ds = image_dataset_from_directory(
    './data/safebooru/', labels=labels, validation_split=0.1, subset='training',
    batch_size=BATCH_SIZE, seed=111)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

print('Loading validation dataset')
val_ds = image_dataset_from_directory(
    './data/safebooru/', labels=labels, validation_split=0.1, subset='validation',
    batch_size=BATCH_SIZE, seed=111)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


vgg = VGG(base)
vgg.summary()
vgg.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
vgg.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
# base_json = base.to_json()
# with open(base_vgg, 'w') as file:
#     file.write(base_json)
# base.save_weights(base_vgg_weights)
