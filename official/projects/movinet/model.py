import tensorflow as tf
import numpy as np

import sys
sys.path.append("/home/ahreumseo/research/violence/datasets/MoViNet-TF/EVD")
from official.vision.configs import video_classification
# from official.projects.movinet.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

from motion_detector import * 
from object_detector import * 

def build_classifier(backbone, num_classes, batch_size, num_frames, resolution, freeze_backbone=False):
    """Builds a classifier on top of a backbone model."""
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        output_states=True)
    model.build([batch_size, num_frames, resolution, resolution, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
            model.layers[-1].trainable = True

    return model


T_CLIPS_TRAIN = 8
T_CLIPS_TEST = 5
N_FRAMES = 25
class CustomModel_modified(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        init_states, image = x

        for j in range(int(N_FRAMES/T_CLIPS_TRAIN)):
            clip = image[:,T_CLIPS_TRAIN*j:T_CLIPS_TRAIN*(j+1),:,:,:]
            
            if j==0:
                # Init states 
                with tf.GradientTape() as tape:
                    pred, states = self({**init_states, 'image': clip}, training=True)
                    loss =  self.compiled_loss(y, pred)
                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

            else:
                # Cumulative loss and gradient
                with tf.GradientTape() as tape:
                    pred, states = self({**states, 'image': clip}, training=True)
                    loss += self.compiled_loss(y, pred)
                gradients += tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        
        x, y = data
        init_states, image = x

        for j in range(int(N_FRAMES/T_CLIPS_TEST)):
            clip = image[:,T_CLIPS_TRAIN*j:T_CLIPS_TRAIN*(j+1),:,:,:]
            
            if j==0:
                # Init states 
                pred, states = self({**init_states, 'image': clip}, training=False)
                loss =  self.compiled_loss(y, pred)
            else:
                # Cumulative loss and gradient
                pred, states = self({**states, 'image': clip}, training=False)
                loss += self.compiled_loss(y, pred)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred)

        return {m.name: m.result() for m in self.metrics}


# class CustomModel_modified(tf.keras.Model):
    
#     # def __init__(self, od_model):
#     #     super(CustomModel_modified, self).__init__()
#     #     self.
#     #     self.od_model = od_model

#     def compile(self, loss, metrics, optimizer, od_model):
#         super(CustomModel_modified, self).compile(optimizer=optimizer, metrics=metrics, loss=loss)
#         self.od_model = od_model 
#         # self.compiled_loss = loss
#         # self.optimizer = optimizer 
#         # self.compiled_metrics = metrics

#     def train_step(self, data):
#         x, y = data
#         init_states, image = x
#         frame_length = N_FRAMES

#         for j in range(int(frame_length/T_CLIPS_TRAIN)):
#             clip = image[:,T_CLIPS_TRAIN*j:T_CLIPS_TRAIN*(j+1),:,:,:]
            
#             if j==0:
#                 # Init states 
#                 with tf.GradientTape() as tape:
#                     pred, states = self({**init_states, 'image': clip}, training=True)
#                     loss =  self.compiled_loss(y, pred)
#                 # Compute gradients
#                 trainable_vars = self.trainable_variables
#                 gradients = tape.gradient(loss, trainable_vars)

#             else:
#                 # Cumulative loss and gradient
#                 with tf.GradientTape() as tape:
#                     pred, states = self({**states, 'image': clip}, training=True)
#                     loss += self.compiled_loss(y, pred)
#                 gradients += tape.gradient(loss, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))


#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, pred)

#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
        
#         x, y = data
#         init_states, image = x[0], x[1]

#         for j in range(int(N_FRAMES/T_CLIPS_TEST)):
#             # clip = image[:,j:j+1,:,:,:]
#             clip = image[:,T_CLIPS_TRAIN*j:T_CLIPS_TRAIN*(j+1),:,:,:]

#             # # IF motion is detected 
#             # if motion_detector(clip):
                
#             #     # If people are detected 
#             #     if object_detector(clip, od_model):
            
#             if j==0:
#                 # Init states 
#                 pred, states = self({**init_states, 'image': clip}, training=False)
#                 loss =  self.compiled_loss(y, pred)
#             else:
#                 # Cumulative loss and gradient
#                 try:
#                     pred, states = self({**states, 'image': clip}, training=False)
#                 except: 
#                     pred, states = self({**init_states, 'image': clip}, training=False)
#                 loss += self.compiled_loss(y, pred)

#             #     else:
#             #         pred = np.array([[0.,1.]], dtype = np.float32)
#             # else: 
#             #     pred = np.array([[0.,1.]], dtype = np.float32)

#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, pred)

#         return {m.name: m.result() for m in self.metrics}






class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # pred, states = self(x, training=True)  # Forward pass
            pred, states = self({**x[0], 'image': x[1]}, training=True)
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
        
        x, y = data
        y_pred, states = self({**x[0], 'image': x[1]}, training=True)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    



def build_model(checkpoint_dir='./movinet_a0_stream', model_id = 'a0', batch_size=1,num_frames=64, resolution=224, pretrain=True):
    
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

    if pretrain:
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_path)
        status.assert_existing_objects_matched()

    model2 = build_classifier(backbone, num_classes=2, batch_size=batch_size, num_frames=num_frames, resolution=resolution, freeze_backbone=False)
    init_states = model.init_states([batch_size, num_frames, resolution, resolution, 3])

    # Build custom model
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

    inputs = {**states_input, 'image': image_input}
    outputs = model2(inputs)
      
    # model3 = CustomModel(inputs, outputs, name='movinet')
    model3 = CustomModel_modified(inputs, outputs)

    return init_states, model3



def build_model_eval(checkpoint_dir="ckpt/rwf-2000_a0_stream", model_id = 'a0', batch_size=1, num_frames=64, resolution=224):

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

    model = build_classifier(backbone, num_classes=2, batch_size=batch_size, num_frames=num_frames, resolution=resolution, freeze_backbone=False)
    init_states = model.init_states([batch_size, num_frames, resolution, resolution, 3])
    
    image_input = tf.keras.layers.Input(shape=[None, None, None, 3],
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

    inputs = {**states_input, 'image': image_input}
    outputs = model(inputs)
      
    model2 = CustomModel(inputs, outputs, name='movinet')
    model2.load_weights(checkpoint_dir)

    return init_states, model2