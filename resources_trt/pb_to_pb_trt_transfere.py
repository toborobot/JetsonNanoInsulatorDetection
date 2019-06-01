import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import os
#from tf_trt_models.detection import build_detection_graph

################################################################################
#need to add object detection libraries and code downloaded From
#https://github.com/tensorflow/models and stored in /home/jnano/ directory
################################################################################
#sys.path.append('/home/jnano/tf/models/research/')
#sys.path.append('/home/jnano/tf/models/research/slim/')
config_path = '/home/jnano/tf/prog10/resources_tf/fine_tuned_model/pipeline.config'
checkpoint_path = '/home/jnano/tf/prog10/resources_tf/fine_tuned_model/model.ckpt'
frozen_graph_path = '/home/jnano/tf/prog10/resources_tf/fine_tuned_model'
output_names = ['num_detections','detection_classes','detection_boxes','detection_scores']
"""
frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,  # path to model’s pipeline.config file
    checkpoint=checkpoint_path,  # path to model.ckpt file
    score_threshold=0.3,
    #iou_threshold=0.5,
    batch_size=1
)
"""
# init graph
# read frozen graph from file
frozen_graph = tf.GraphDef()
with open(os.path.join(frozen_graph_path, 'frozen_inference_graph.pb'), 'rb') as f:
    frozen_graph.ParseFromString(f.read())

"""
link https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

def create_inference_graph(input_graph_def,
                           outputs,
                           max_batch_size=1,
                           max_workspace_size_bytes=2 << 20,
                           precision_mode="fp32",
                           minimum_segment_size=3,
                           is_dynamic_op=False,
                           maximum_cached_engines=1,
                           cached_engine_batch_sizes=None
                           use_calibration=True,
                           rewriter_config=None,
                           input_saved_model_dir=None,
                           input_saved_model_tags=None,
                           output_saved_model_dir=None,
                           session_config=None):

Where:
-input_graph_def
This parameter is the GraphDef object that contains the model to be transformed.
-outputs
This parameter lists the output nodes in the graph. Tensors which are not marked
 as outputs are considered to be transient values that may be optimized away by
 the builder.
-max_batch_size
This parameter is the maximum batch size that specifies the batch size for which
 TensorRT will optimize. At runtime, a smaller batch size may be chosen. At runtime,
 larger batch size is not supported.
-max_workspace_size_bytes
TensorRT operators often require temporary workspace. This parameter limits the
 maximum size that any layer in the network can use. If insufficient scratch is
  provided, it is possible that TensorRT may not be able to find an implementation
  for a given layer.
-precision_mode
TF-TRT only supports models trained in FP32, in other words all the weights of
 the model should be stored in FP32 precision. That being said, TensorRT can
 convert tensors and weights to lower precisions during the optimization.
 The precision_mode parameter sets the precision mode; which can be one of
 fp32, fp16, or int8. Precision lower than FP32, meaning FP16 and INT8, would
  improve the performance of inference. The FP16 mode uses Tensor Cores or half
  precision hardware instructions, if possible. The INT8 precision mode uses
   integer hardware instructions.
-minimum_segment_size
This parameter determines the minimum number of TensorFlow nodes in a TensorRT
engine, which means the TensorFlow subgraphs that have fewer nodes than this
number will not be converted to TensorRT. Therefore, in general smaller numbers
 such as 5 are preferred. This can also be used to change the minimum number of
  nodes in the optimized INT8 engines to change the final optimized graph to
  fine tune result accuracy.
-is_dynamic_op
If this parameter is set to True, the conversion and building the TensorRT
engines will happen during the runtime, which would be necessary if there are
tensors in the graphs with unknown initial shapes or dynamic shapes. For more
information, see index.html#static-dynamic-mode.
Note: Conversion during runtime may increase the latency, depending on your
model and how you use it.
-maximum_cached_engines
In dynamic mode, this sets the maximum number of cached TensorRT engines per
TRTEngineOp. For more information, see index.html#static-dynamic-mode.
-cached_engine_batch_sizes
The list of batch sizes used to create cached engines, only used when
is_dynamic_op is True. The length of the list should be smaller than
maximum_cached_engines, and the dynamic TensorRT op will use this list to
determine the batch sizes of the cached engines, instead of making the decision
 while in progress. This is useful when we know the most common batch size(s)
 the application is going to generate.
-cached_engine_batches
The batch sizes used to pre-create cached engines.
-use_calibration
This argument is ignored if precision_mode is not INT8.
If set to True, a calibration graph will be created to calibrate the missing
ranges. The calibration graph must be converted to an inference graph using
calib_graph_to_infer_graph() after running calibration.
If set to False, quantization nodes will be expected for every tensor in the
graph (excluding those which will be fused). If a range is missing, an error will occur.
Note: Accuracy may be negatively affected if there is a mismatch between which
tensors TensorRT quantizes and which tensors were trained with fake quantization.
-rewriter_config
A RewriterConfig proto to append the TensorRTOptimizer to. If None, it will
create one with default settings.
-input_saved_model_dir
The directory to load the SavedModel containing the input graph to transform.
Used only when input_graph_def is None.
-input_saved_model_tags
A list of tags used to identify the MetaGraphDef of the SavedModel to load.
-output_saved_model_dir
If not None, construct a SavedModel using the returned GraphDef and save it to
the specified directory. This option only works when the input graph is loaded
from a SavedModel, in other words, when input_saved_model_dir is specified and
input_graph_def is None.
-session_config
The ConfigProto used to create a Session. If not specified, a default ConfigProto
will be used.

Returns:
New GraphDef with TRTEngineOps placed in graph replacing subgraphs.
Raises:
ValueError: If the provided precision mode is invalid.
RuntimeError: If the returned status message is malformed.
"""

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1<<25,
    precision_mode='FP16',
    minimum_segment_size=5)

with open('/home/jnano/tf/prog10/resources_trt/trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
