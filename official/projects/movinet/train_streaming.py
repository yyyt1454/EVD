import tensorflow_datasets as tfds
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
import os
from tqdm import tqdm

# os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/")
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
print(tf.config.list_physical_devices('GPU'))


batch_size = 1
num_frames = 64
# frame_stride = 0
resolution = 224
num_epochs = 50
model_id = 'a0'

tf.keras.backend.clear_session()

# Create backbone and model.
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
backbone = movinet.Movinet(
	model_id=model_id,
	causal=True,
	conv_type='2plus1d',
	se_type='2plus3d',
	activation='hard_swish',
	gating_activation='hard_sigmoid',
	use_positional_encoding=use_positional_encoding,
	use_external_states=True,
)


model = movinet_model.MovinetClassifier(
	backbone,
	num_classes=600,
	output_states=True)
model.build([1, 1, 1, 1, 3])

checkpoint_dir = 'movinet_a0_stream'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(backbone, num_classes, freeze_backbone=False):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes,
      output_states=True)
  model.build([batch_size, num_frames, resolution, resolution, 3])
#   model.build([1, 64, 224, 224, 3])

  if freeze_backbone:
    for layer in model.layers[:-1]:
      layer.trainable = False
    model.layers[-1].trainable = True

  return model



model2 = build_classifier(backbone, num_classes=2, freeze_backbone=False)
init_states = model.init_states([batch_size, num_frames, resolution, resolution, 3])

## Data generator
train_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/train', 
                                batch_size=batch_size, 
                                data_augmentation=True,
                                init_states = init_states)

val_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/val',
                              batch_size=batch_size, 
                              data_augmentation=False,
                              init_states = init_states)



# # Create a movinet classifier using this backbone.
# model = movinet_model.MovinetClassifier(
# 	backbone,
# 	num_classes=600,
# 	output_states=True)



# movinet_hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/3'

# movinet_hub_model = hub.KerasLayer(movinet_hub_url, trainable=True)

# pretrained_weights = {w.name: w for w in movinet_hub_model.weights}

# model_weights = {w.name: w for w in model.weights}

# for name in pretrained_weights:
# # 	model_weights[name].assign(pretrained_weights[name])
#     for w in model.weights:
#         if name == w.name:
#             w.assign(pretrained_weights[name])
#             break
#         else:
#             continue


# model = movinet_model.MovinetClassifier(
# 	backbone=backbone,
# 	num_classes=2,
# 	output_states=True)


# model.build([batch_size, num_frames, resolution, resolution, 3])
# init_states = model.init_states([batch_size, num_frames, resolution, resolution, 3])

# Input layer for the frame sequence
image_input = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

state_shapes = {
    name: ([s if s > 0 else None for s in state.shape], state.dtype)
    for name, state in init_states.items()
}

states_input = {
    name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
    for name, (shape, dtype) in state_shapes.items()
}

# Wrap the Movinet model in a Keras model so that it can be finetuned.
inputs = {**states_input, 'image': image_input}
outputs = model2(inputs)

# This custom training step ignores the updated states during training as they are only important during inference.
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # pred, states = self({**init_states, 'image': x}, training=True)  # Forward pass
            pred, states = self(x, training=True)  # Forward pass
            # print(pred)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        
        # Unpack the data
        x, y = data
        # Compute predictions
        # y_pred, states = self({**init_states, 'image': x}, training=True)
        y_pred, states = self(x, training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

model2 = CustomModel(inputs, outputs, name='movinet')

for layer in model2.layers[:-1]:
	layer.trainable = True
model2.layers[-1].trainable = True


## Training methods
loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# metrics = ['accuracy']
metrics = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=10,
)
optimizer = tf.keras.optimizers.RMSprop(
    initial_learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)
callbacks = [
    tf.keras.callbacks.TensorBoard(),
    tf.keras.callbacks.ModelCheckpoint(filepath='./rwf-2000_a0_stream', monitor="val_movinet_classifier_1_accuracy", mode="max", save_freq="epoch", save_best_only=True)
]

model2.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)

model2.fit(train_generator, validation_data=val_generator, epochs=50,callbacks=callbacks)


