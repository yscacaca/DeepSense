# Preparing a TF model for usage in Android
# By Omid Alemi - Jan 2017
# Works with TF r1.0

import sys
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_SAVE_DIR = 'android_model_saver'
MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME+'.pbtxt')
checkpoint_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME+'.ckpt')
input_saver_def_path = ""
input_binary = False
output_node_names = "O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = os.path.join(MODEL_SAVE_DIR,'frozen_'+MODEL_NAME+'.pb')
output_optimized_graph_name = os.path.join(MODEL_SAVE_DIR,'optimized_'+MODEL_NAME+'.pb')
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference # The optimized pb file does not run properly on Android now

# input_graph_def = tf.GraphDef()
# with tf.gfile.Open(output_frozen_graph_name, "r") as f:
#     data = f.read()
#     input_graph_def.ParseFromString(data)

# output_graph_def = optimize_for_inference_lib.optimize_for_inference(
#         input_graph_def,
#         ["I"], # an array of the input node(s)
#         ["O"], # an array of output nodes
#         tf.float32.as_datatype_enum)

# # Save the optimized graph

# f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
# f.write(output_graph_def.SerializeToString())

# # tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)                    
