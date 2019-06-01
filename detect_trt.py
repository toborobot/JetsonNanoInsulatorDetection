#2019-05-19 for detection and classification with TensorRT optimized graph (directory resources_trt)
#need specially prepared graph with pb_to_pb_trt_transfere.py script
#very slow loaded to memory ( about 3 min)
import tensorflow as tf
import cv2
import os
import time

init_time = time.time()

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


PATH_TO_TEST_IMAGES_DIR = './test/images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '8{}.jpg'.format(i)) for i in range(5, 10) ]

# The TensorRT frozen inference graph
pb_fname = "./resources_trt/trt_graph.pb"
trt_graph = get_frozen_graph(pb_fname)

input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

print('time of loading:'+str(time.time()-init_time))

for counter_ in range(0,10): # average detection time 0.087 per one image ~ 11FPS
    for image_path in TEST_IMAGE_PATHS:
        cycle_time = time.time()
        image = cv2.imread(image_path)
        #image = cv2.resize(image, (300, 300))

        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
            tf_input: image[None, ...]
        })
        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])

        print('time of inference:'+str(time.time()-cycle_time))
        print("detected " + str(num_detections))
