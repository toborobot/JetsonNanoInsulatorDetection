# Import TensorFlow and TensorRT
# from https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
# Inference with TF-TRT frozen graph workflow:

frozen_graph_path = '/home/jnano/tf/prog10/resources_tf/fine_tuned_model/frozen_inference_graph.pb'
output_names = ['num_detections','detection_classes','detection_boxes','detection_scores']

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1<<25,
            precision_mode='FP16')
        # Import the TensorRT graph into a new graph and run:
with open('/home/jnano/tf/prog10/resources_trt/trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
