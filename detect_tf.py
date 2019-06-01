#detect and classify image program test for jetson nano
#python PIL realisation and direct tensorflow graph not TensorRT optimized
#2019-05-25
#checking for library installed with tf_trt_models installation

import numpy as np
import os
import sys
import tensorflow as tf
# with PIL time to process1 image is about 0,25 sec
#from PIL import Image # choose library for image processing
# with cv2 time to process 1 image is about 0,07 sec
import cv2  # coose which library will process image
import base64
import time

################################################################################
#need to add object detection libraries and code downloaded From
#https://github.com/tensorflow/models and stored in /home/jnano/ directory
# this is important if not installed tt_trt_models with install.sh script
#because it insltall all PATHs with installation and tensorflow models located in
#tf_trt_models/third_party/models directory
################################################################################
#sys.path.append('/home/jnano/tf_trt_models/third_party/models/research/')
#sys.path.append('/home/jnano/tf_trt_models/third_part/models/research/slim/')

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

################################################################################
#load image to numpy array procedure for wirking with PIL image processing
################################################################################
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

################################################################################
#MAIN procedure
################################################################################
# start timing of processes
time_start = time.time()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './resources_tf/fine_tuned_model' + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('./resources_tf/data/', 'object-detection.pbtxt')
#number ofclassesfor classification
NUM_CLASSES = 1

#for local testing
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = './test/images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '8{}.jpg'.format(i)) for i in range(5, 10) ]



# init graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#creat labels
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#print(categories)
# return [{'name': 'insulator', 'id': 1}]
#print(category_index)
#return {1: {'name': 'insulator', 'id': 1}}

#init graph all the vars
with detection_graph.as_default():
  with tf.Session() as sess:
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    print(time.time()-time_start)

    for counter_ in range(0,10): # average detection time 0.067 per one image ~ 15FPS
        for image_path in TEST_IMAGE_PATHS:
          #this part is for PIL image processing
          """
          image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          #end of part for PIL image processing
          """
          #this part is for cv2 image processing

          in_file = open(image_path, "rb")
          data = in_file.read()
          in_file.close()
          encoded_string = base64.standard_b64encode(data)
          file_string = str(encoded_string, 'ascii', 'ignore')

          file_bytes = np.asarray(bytearray(base64.b64decode(file_string)), dtype=np.uint8)
          image_ = cv2.imdecode(file_bytes, 1)
          #convertcolor from BGR to RGB
          image_np = cv2.cvtColor(image_, 4)

          # Actual detection.
          time_cycle = time.time()

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
          print('time of inference:'+str(time.time() - time_cycle))
          print("detected "+str(output_dict['num_detections']))
          #print(output_dict['detection_boxes'])
          #print(output_dict['detection_classes'])
          #print(output_dict['detection_scores'])
          #print(output_dict['detection_classes'])
          #print(output_dict.get('detection_masks'))

          # Visualization of the results of a detection.
          """
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=3)

          vis_util.save_image_array_as_png(image_np,'./test/images/result.png')
          #make result png string from image numpy array
          #result_string = vis_util.encode_image_array_as_png_str(image_np)
          #print(result_string)
          """

print('thats all folks')
#sys.exit(0)
