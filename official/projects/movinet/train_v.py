import tensorflow_datasets as tfds
import tensorflow as tf

import sys
sys.path.append("/home/ahreumseo/research/violence/datasets/MoViNet-TF/models")
from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

from dataloader import *
import os

# os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/")
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
print(tf.config.list_physical_devices('GPU'))


batch_size = 1
num_frames = 64
frame_stride = 0
resolution = 224
num_epochs = 50
model_id = 'a0'

train_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/train', 
                                batch_size=batch_size, 
                                data_augmentation=True)

val_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/val',
                              batch_size=batch_size, 
                              data_augmentation=False)


tf.keras.backend.clear_session()


# num_classes 600짜리를 만들어서 checkpoint load
backbone = movinet.Movinet(model_id=model_id)
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([1, 1, 1, 1, 3])

checkpoint_dir = 'movinet_a4_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Checkpoin load된 backbone 떼와서 2개짜리 classifier 붙이기
def build_classifier(backbone, num_classes, freeze_backbone=False):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  if freeze_backbone:
    for layer in model.layers[:-1]:
      layer.trainable = False
    model.layers[-1].trainable = True

  return model

# Wrap the backbone with a new classifier to create a new classifier head
# with num_classes outputs (101 classes for UCF101).
# Freeze all layers except for the final classifier head.
model = build_classifier(backbone, num_classes=2, freeze_backbone=False)


loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=10,
)
optimizer = tf.keras.optimizers.RMSprop(
    initial_learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
# optimizer = tf.keras.optimizers.Adam(initial_learning_rate)


callbacks = [
    tf.keras.callbacks.TensorBoard(),
    tf.keras.callbacks.ModelCheckpoint(filepath='./rwf-2000_a0_scratch', monitor="val_accuracy", mode="max", save_freq="epoch", save_best_only=True)
]

# model = tf.keras.models.load_model('./rwf-2000_a4')
model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)

results = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs,
#     steps_per_epoch=1,
#     validation_steps=1,
    callbacks=callbacks,
#     validation_freq=1,
    workers=1)
