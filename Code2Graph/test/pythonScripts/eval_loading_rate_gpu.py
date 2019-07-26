# -*- coding: utf-8 -*-
'''
export LD_LIBRARY_PATH=./OpenALPR_python:$LD_LIBRARY_PATH
python object_detection/eval_loading_rate.py --video_file=/media/hu/186E61E86E61BF5E/video-analysis/videos/loading63_8.mp4\
 --detect_model_path=/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/models/model/frozen_inference_graph/detection_frozen_inference_graph.pb\
 --detect_labels_path=/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/data/cad_label_map.pbtxt\
 --classify_model_path=/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/models/model/frozen_inference_graph/classification_inception_v4_freeze.pb\
 --classify_labels_path=/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/models/model/frozen_inference_graph/classification_inception_v4_freeze.label
'''
from openalpr import Alpr

from utils import label_map_util
from utils import visualization_utils as vis_util

import os
import cv2
import numpy as np
import tensorflow as tf


#########################################
# Options
#########################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('video_file', '', 'The name of the input video file.')
tf.app.flags.DEFINE_string('gpus', '0', 'GPUs to use for testing.')
tf.app.flags.DEFINE_string('detect_model_path', '', 'The frozen pb file containing network structure and parameters.')
tf.app.flags.DEFINE_string('detect_labels_path', '', 'path to label map')
tf.app.flags.DEFINE_string('classify_model_path', '', 'The frozen pb file containing network structure and parameters.')
tf.app.flags.DEFINE_string('classify_labels_path', '', 'path to label map')


#########################################
# Subroutines
#########################################
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_path=None):
    if not label_path:
      tf.logging.fatal('please specify the label file.')
      return
    self.node_lookup = self.load(label_path)

  def load(self, label_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(label_path):
      tf.logging.fatal('File does not exist %s', label_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
    id_to_human = {}
    for line in proto_as_ascii_lines:
      if line.find(':') < 0:
        continue
      _id, human = line.rstrip('\n').split(':')
      id_to_human[int(_id)] = human

    return id_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def show_detection(image, score, window_wd, window_ht, extra_region):
  image = cv2.resize(image, (window_wd, window_ht))
  canvas = cv2.copyMakeBorder(image, 0, extra_region, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

  # rect_width = int(score * window_wd)
  # rect_color = int(score*255) 红 橙 黄 绿 青 蓝 紫
  # cv2.rectangle(canvas, (0,window_ht), (rect_width, window_ht+extra_region), (0, 255-rect_color, rect_color), -1)  # -1 means filled

  return canvas

class Detect_inference(object):
  def __init__(self, pb_path):
    self.graph = self.load_graph(pb_path)
    print("Successfully loaded detect_model from disk file %s" % pb_path)
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
    print("Successfully loaded detect_model from disk file %s" % pb_path)
    self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
    self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
    self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
    self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
    self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
    self.boxes = None
    self.scores = None
    self.classes = None
    self.num = None
  def load_graph(self, pb_path):
    detect_graph = tf.Graph()
    with detect_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(pb_path, "rb") as f:
        serialized_graph = f.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detect_graph
  def predict(self, np_image):
    np_image_expanded = np.expand_dims(np_image, axis=0) # np.ndarray, (1, height, width, 3), np.uint8
    (self.boxes, self.scores, self.classes, self.num) = self.sess.run([self.detection_boxes, self.detection_scores,
          self.detection_classes, self.num_detections], feed_dict={self.image_tensor : np_image_expanded})
    return self.boxes, self.scores, self.classes, self.num

class LR_inference(object):
  def __init__(self, pb_path):
    self.graph = self.load_graph(pb_path)
    self.sess_for_np_image = tf.Session()
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
    print("Successfully loaded LR_classification_model from disk file %s" % pb_path)
    #self.image_tensor = self.graph.get_tensor_by_name('input:0')
    self.classification_scores = self.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
    self.scores = None
    self.input_height = 299
    self.input_width = 299
    self.image_tensor = None
  def load_graph(self, pb_path):
    classify_graph = tf.Graph()
    with classify_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(pb_path, 'rb') as f:
        serialized_graph = f.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return classify_graph
  def predict(self, np_image):
    self.image_tensor = self.__preprocess_for_eval(np_image, self.input_height, self.input_width)
    np_image = self.sess_for_np_image.run(self.image_tensor)
    #np_image = self.image_tensor.eval(session = tf.Session())
    np_image_expanded = np.expand_dims(np_image, axis=0) # np.ndarray, (1, height, width, 3), np.uint8
    self.scores = self.sess.run(self.classification_scores, feed_dict={'input:0' : np_image_expanded})
    return self.scores
  def __preprocess_for_eval(self, image, height, width, central_fraction=0.875, scope=None):
    with tf.name_scope(scope, '', [image, height, width]) as scope:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32) # 0.0~1.0
      image = tf.image.central_crop(image, central_fraction=central_fraction)
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
      image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
    return image


#########################################
# Main Flow
#########################################
def main(_):
  # loading detection label map
  # PATH_TO_LABELS = '/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/data/cad_label_map.pbtxt'
  NUM_CLASSES = 3
  label_map = label_map_util.load_labelmap(FLAGS.detect_labels_path)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  '''
  {1: {'id': 1, 'name': 'person'},
   2: {'id': 2, 'name': 'vanRear'},
   3: {'id': 3, 'name': 'plate'}}
  '''

  # loading classification label map
  node_lookup = NodeLookup(FLAGS.classify_labels_path)
  # human_string = node_lookup.id_to_string(node_id)

  if not FLAGS.video_file:
    raise ValueError('You must supply the video file name (--video_file)')

  ############################################################
  print ("initializing...")
  ############################################################
  detect_inference = Detect_inference(FLAGS.detect_model_path)
  lr_inference = LR_inference(FLAGS.classify_model_path)
  alpr = Alpr('cn', 'OpenALPR_python/openalpr.conf', 'OpenALPR_python/runtime_data')
  if not alpr.is_loaded():
    print('!!! Error loading OpenALPR')
  else:
    print('Using OpenALPR', alpr.get_version())
    # image = cv2.imread("../OpenALPR_python/5.png")
    # best = alpr.recognize_plate(image)
    # alpr.unload()

  
  ''' ##################test models########################
  img_to_detect = cv2.imread('/media/hu/186E61E86E61BF5E/video-analysis/test_images/35.jpg')
  img_to_classify = cv2.imread('/home/hu/Downloads/tensorflow-models/research/VOCdevkit/VOC2012/models/model/frozen_inference_graph/loadingRate/tesdata/93.png')
  (boxes, scores, classes, num) = detect_inference.predict(img_to_detect)
  classify_predictions = lr_inference.predict(img_to_classify)
  vis_util.visualize_boxes_and_labels_on_image_array(
          img_to_detect,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
  cv2.imshow('test', img_to_detect)
  cv2.waitKey(0)

  classify_predictions = np.squeeze(classify_predictions)

  top_k = classify_predictions.argsort()[-2:][::-1]
  top_names = []
  for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    top_names.append(human_string)
    score = classify_predictions[node_id]
    print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    print(classify_predictions, top_k, top_names)
  '''

  
  ############################################################
  print ("program launched...")
  ############################################################
  displayWidth = 1280
  extra_region = 100

  cap = cv2.VideoCapture(FLAGS.video_file)
  fps = cap.get(cv2.CAP_PROP_FPS)
  wd = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  left_bound = 1/3
  right_bound = 2/3
  # bottom_bound = None

  ratiorgb = ht / wd
  displayHeight = int(displayWidth*ratiorgb)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  outfile = cv2.VideoWriter('output63_8_0.avi', fourcc, fps, (displayWidth, displayHeight + extra_region))
  #cv2.namedWindow('test')

  state = -1 # -1 : unknown, 0 : noVan, 1 : van
  plate = None
  loadingRate = None
  plateNumber = 0
  vanRearCounter = 0
  noVanRearCounter = 0
  confirmVanRear = 5
  confirmNoVanRear_for_noVan = int(fps)
  confirmNoVanRear_for_van = int(2 * fps)
  vanRears = []
  plates = []
  min_score_thresh = 0.5
  x_center = 0
  van_is_moving = False
  hasVanRear = False
  bestPlate = None

  fontFace = cv2.FONT_HERSHEY_COMPLEX
  duration_displaying_loading_rate = int(10*fps)
  counter_for_displaying = 0

  while True:
    ret, image_np = cap.read()
    if not ret:
      break
    (boxes, scores, classes, num) = detect_inference.predict(image_np)
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    if van_is_moving:
      hasVanRear = False
      for i in range(int(num[0])):
        if scores[i] >= min_score_thresh:
          if classes[i] == 2:
            x_center = (boxes[i][1] + boxes[i][3])/2
            if x_center >= left_bound and x_center <= right_bound:
              hasVanRear = True
              # vanRearCounter += 1
              # vanRears.append(image_np[int(boxes[i][0]*ht):int(boxes[i][2]*ht)+1, int(boxes[i][1]*wd):int(boxes[i][3]*wd)+1])
              for j in range(int(num[0])):
                if scores[j] >= min_score_thresh:
                  if classes[j] == 3 and boxes[j][0] >= boxes[i][0] and boxes[j][1] >= boxes[i][1] and boxes[j][2] <= boxes[i][2] and boxes[j][3] <= boxes[i][3]:
                    plates.append(image_np[int(boxes[j][0]*ht):int(boxes[j][2]*ht)+1, int(boxes[j][1]*wd):int(boxes[j][3]*wd)+1])
                    vanRears.append(image_np[int(boxes[i][0]*ht):int(boxes[j][0]*ht)+1, int(boxes[i][1]*wd):int(boxes[i][3]*wd)+1])
                    #cv2.imshow('plate', plates[-1])
                    #cv2.imshow('vanRear', vanRears[-1])
                    #cv2.waitKey(1)
                    break # already found plate, then break the loop
                else:
                  break
              break
        else:
          break
      if not hasVanRear:
        noVanRearCounter += 1
        if noVanRearCounter == confirmNoVanRear_for_noVan:# and plates != []: 
          sum1 = vanRears[0].shape[1] + vanRears[1].shape[1] + vanRears[2].shape[1]
          sum2 = vanRears[-1].shape[1] + vanRears[-2].shape[1] + vanRears[-3].shape[1]
          if sum1 > sum2:
            van_is_moving = False # noVan confirmed
            state = 0 # noVan
            noVanRearCounter = 0
            counter_for_displaying = 0
            print('-----------------------noVan confirmed-------------------------------')
            # evaluate loading rate and plate number
            plateNumber = len(plates)
            bestPlate = alpr.recognize_plate(plates[int(0.1*plateNumber)+1])
            plate = bestPlate['Plate']
            #classify_predictions = lr_inference.predict(img_to_classify)
            classify_predictions = lr_inference.predict(vanRears[int(0.05*plateNumber)+1])
            classify_predictions = np.squeeze(classify_predictions)
            
            top_k = classify_predictions.argsort()[-1:][::-1]
            top_names = []
            for node_id in top_k:
              human_string = node_lookup.id_to_string(node_id)
              top_names.append(human_string)
              score = classify_predictions[node_id]
              loadingRate = human_string
            vanRears.clear()
            plates.clear()
          else:
            pass
        elif noVanRearCounter == confirmNoVanRear_for_van:# and plates != []:
          #sum1 = vanRears[5].shape[1] + vanRears[6].shape[1] + vanRears[7].shape[1]
          #sum2 = vanRears[-6].shape[1] + vanRears[-7].shape[1] + vanRears[-8].shape[1]
          #if sum1 < sum2:
          van_is_moving = False # van confirmed
          state = 1 # van
          noVanRearCounter = 0
          counter_for_displaying = 0
          print('+++++++++++++++++++++++++++van confirmed++++++++++++++++++++++++++++++++++')
          # evaluate loading rate and plate number
          plateNumber = len(plates)
          cv2.imwrite('plate.png', plates[int(0.9*plateNumber)])
          bestPlate = alpr.recognize_plate(plates[int(0.9*plateNumber)])
          plate = bestPlate['Plate']
          classify_predictions = lr_inference.predict(vanRears[int(0.95*plateNumber)])
          classify_predictions = np.squeeze(classify_predictions)

          top_k = classify_predictions.argsort()[-1:][::-1]
          top_names = []
          for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = classify_predictions[node_id]
            loadingRate = human_string
          vanRears.clear()
          plates.clear()
      else:
        noVanRearCounter = 0 # van is still moving
    else:
      hasVanRear = False
      for i in range(int(num[0])):
        if scores[i] >= min_score_thresh:
          if classes[i] == 2:
            x_center = (boxes[i][1] + boxes[i][3])/2
            if x_center >= left_bound and x_center <= right_bound:
              hasVanRear = True
              vanRearCounter += 1
              # vanRears.append(image_np[int(boxes[i][0]*ht):int(boxes[i][2]*ht)+1, int(boxes[i][1]*wd):int(boxes[i][3]*wd)+1])
              for j in range(int(num[0])):
                if scores[j] >= min_score_thresh:
                  if classes[j] == 3 and boxes[j][0] >= boxes[i][0] and boxes[j][1] >= boxes[i][1] and boxes[j][2] <= boxes[i][2] and boxes[j][3] <= boxes[i][3]:
                    plates.append(image_np[int(boxes[j][0]*ht):int(boxes[j][2]*ht)+1, int(boxes[j][1]*wd):int(boxes[j][3]*wd)+1])
                    vanRears.append(image_np[int(boxes[i][0]*ht):int(boxes[j][0]*ht)+1, int(boxes[i][1]*wd):int(boxes[i][3]*wd)+1])
                    #cv2.imshow('plate', plates[-1])
                    #cv2.imshow('vanRear', vanRears[-1])
                    #cv2.waitKey(1)
                    break # already found plate, then break the loop
                else:
                  break # confidence too low, then break the loop
              break # already found vanRear, then break the loop
        else:
          break # confidence too low, then break the loop
    
      if not hasVanRear:
        vanRearCounter = 0
        vanRears.clear()
        plates.clear()
        # van_is_moving = False
      elif vanRearCounter == confirmVanRear:
        if plates != []:
          van_is_moving = True
          vanRearCounter = 0
          print('===================van is moving======================')
        else:
          vanRearCounter = 0
      else:
        #van_is_moving = False # just continue to accumulate evidence for van_is_moving to be True
        pass

    
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          boxes,
          classes,
          scores,
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=None,
          min_score_thresh=.5,
          line_thickness=2)
    image_np = cv2.resize(image_np, (displayWidth, displayHeight))
    canvas = cv2.copyMakeBorder(image_np, 0, extra_region, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    if van_is_moving:
      cv2.putText(canvas, 'a van is moving', (int(0.1*displayWidth), int(displayHeight+0.7*extra_region)), fontFace, 2, (255, 255, 255), 2, 8)
    elif state == 0:
      # noVan confirmed
      # duration_displaying_loading_rate = int(10*fps)
      # counter_for_displaying = 0
      if counter_for_displaying < duration_displaying_loading_rate:
        # display
        cv2.putText(canvas, 'the van has departured', (int(0.1*displayWidth), int(displayHeight+0.3*extra_region)), fontFace, 1, (255,255,255),2,8)
        cv2.putText(canvas, u'loadingRate: %s, plateNumber: %s' % (loadingRate, plate), (int(0.1*displayWidth), int(displayHeight+0.8*extra_region)), fontFace, 1, (255, 255, 255), 2, 8)
        counter_for_displaying += 1
      else:
        # stop displaying
        # counter_for_displaying = 0
        pass
    elif state == 1:
      # van confirmed
      if counter_for_displaying < duration_displaying_loading_rate:
        # display
        cv2.putText(canvas, 'the van is in dock', (int(0.1*displayWidth), int(displayHeight+0.3*extra_region)), fontFace, 1, (255,255,255),2,8)
        cv2.putText(canvas, u'loadingRate: %s, plateNumber: %s' % (loadingRate, plate), (int(0.1*displayWidth), int(displayHeight+0.8*extra_region)), fontFace, 1, (255, 255, 255), 2, 8)
        counter_for_displaying += 1
      else:
        # stop displaying
        # counter_for_displaying = 0
        pass
    outfile.write(canvas)
    #cv2.imshow('test', canvas)
    #if cv2.waitKey(1) == ord('q'):
      #break
  
  cap.release()
  outfile.release()
  alpr.unload()
  cv2.destroyAllWindows()

if __name__ == '__main__': 
  tf.app.run()
