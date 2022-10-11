# import tensorflow_datasets as tfds
import tensorflow as tf

import sys
sys.path.append("/home/ahreumseo/research/violence/datasets/MoViNet-TF/EVD")
from official.vision.configs import video_classification
# from official.projects.movinet.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

from dataloader import *
from model import *
import os
from tqdm import tqdm

# os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/")
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
print(tf.config.list_physical_devices('GPU'))

# Config
batch_size = 1
num_frames = 64
# frame_stride = 0
resolution = 224
num_epochs = 50
model_id = 'a0'
checkpoint_dir = 'movinet_a0_stream'


tf.keras.backend.clear_session()

# Model
init_states, model = build_model(checkpoint_dir, model_id, batch_size, num_frames, resolution)

## Data generator
train_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/train', 
                                batch_size=batch_size, 
                                data_augmentation=True,
                                init_states = init_states)

val_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/val',
                              batch_size=batch_size, 
                              data_augmentation=False,
                              init_states = init_states)

## Training methods
loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=10,
)
optimizer = tf.keras.optimizers.RMSprop(
    initial_learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

checkpoint_path = "./ckpt/rwf-2000_a0_stream"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    tf.keras.callbacks.TensorBoard(),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                       monitor="val_movinet_classifier_1_accuracy", 
                                       mode="max", save_freq="epoch", 
                                       save_best_only=True,
                                      save_weights_only=True)
]

model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)

model.fit(train_generator, validation_data=val_generator, epochs=50,callbacks=callbacks)






