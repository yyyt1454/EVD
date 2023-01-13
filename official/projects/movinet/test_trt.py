import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time

trt_path = './models/efficientdet-d0.trt'

# Some helper functions
def get_func_from_saved_model(saved_model_dir):
   saved_model_loaded = tf.saved_model.load(
       saved_model_dir, tags=[tag_constants.SERVING])
   graph_func = saved_model_loaded.signatures[
       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
   return graph_func, saved_model_loaded
 
 
def get_random_input(batch_size, seq_length):
   # Generate random input data
   image = tf.convert_to_tensor(np.ones((batch_size, seq_length, seq_length, 3), dtype=np.float16))
   return image
 
 
# Get a random input tensor
input_tensor = get_random_input(1, 512)
 
# Specify the output tensor interested in. This output is the 'classifier'
trt_func, _ = get_func_from_saved_model(trt_path)
 
## Let's run some inferences!
for i in range(0, 10):
   start = time.time()
   preds = trt_func(input_tensor)
   end=time.time()
   print(f'inference time: {end-start}')

