import os
import tensorflow as tf

from dataloader import *
from model import *

from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model



# os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/")


batch_size = 1
num_frames = 64
frame_stride = 0
resolution = 224

# Model
init_states, model = build_model_eval(checkpoint_dir="ckpt/rwf-2000_a0_stream",
                                    model_id='a0', 
                                    batch_size=batch_size, 
                                    num_frames=num_frames, 
                                    resolution=resolution)




loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=10)
optimizer = tf.keras.optimizers.RMSprop(initial_learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

model.compile(loss = loss_obj, metrics=['accuracy'], optimizer=optimizer)

# RWF-2000
val_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/val',
                              batch_size=batch_size, 
                              data_augmentation=False,
                             shuffle=False,
                             init_states = init_states)

# Surveillance Fight
val_generator2 = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/Surveillance_Fight/npy_224',
                              batch_size=batch_size, 
                              data_augmentation=False,
                             shuffle=False,
                             init_states = init_states)

model.fit(val_generator, validation_data=val_generator, epochs=3)
model.evaluate(val_generator)

model.evaluate(val_generator2)