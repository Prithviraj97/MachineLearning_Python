import numpy as np
import pandas as pd

def calculate_iou(gt_box, det_box):
  """
  Calculates Intersection-over-Union (IoU) between two bounding boxes.

  Args:
      gt_box: Ground truth bounding box (list of [y_min, x_min, y_max, x_max] coordinates).
      det_box: Detected bounding box (list of [y_min, x_min, y_max, x_max] coordinates).

  Returns:
      IoU value (float) between 0 and 1. 
      IoU of 0.5 is considered as threshold to classify detection as true positive.
  """
  # Calculate area of overlap and area of union
  ymin_intersection = np.maximum(gt_box[:,0], det_box[0])
  xmin_intersection = np.maximum(gt_box[:,1], det_box[1])
  ymax_intersection = np.minimum(gt_box[:,2], det_box[2])
  xmax_intersection = np.minimum(gt_box[:,3], det_box[3])

  area_intersection = np.maximum(0, xmax_intersection - xmin_intersection) * np.maximum(0, ymax_intersection - ymin_intersection)
  area_gt = (gt_box[:,2] - gt_box[:,0]) * (gt_box[:,3] - gt_box[:,1])
  area_det = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
  area_union = area_gt + area_det - area_intersection

  # Calculate IoU
  iou = area_intersection / area_union  
  return iou

def evaluate_model(gt_bbox, pred_bbox, iou_threshold=0.5):
  
  """
  Evaluates a model's performance using Intersection over Union (IoU).
  
  Args:
    gt_bbox: A list of lists containing the ground truth bounding boxes for each image in the dataset. 
             Each bounding box should be a list of four integers representing [ymin, xmin, ymax, xmax].
    pred_bbox: A list of predicted bounding boxes in the same format as `gt_bbox`.
    
  Kwargs:
    iou_threshold: The minimum IoU required for a prediction to count as correct. Default is 0.5.

  Returns:
    - precision: Precision of the detection.
    - recall: Recall of the detection.
    - f1_score: F1 score of the detection.
  """
  #if the list of ground truth bounding box or prediction bounding box is empty then we can't calculate the precision, recall and f1.
  if len(gt_bbox) == 0 or len(pred_bbox) == 0:
        return 0, 0, 0
  
  gt_bbox = np.array(gt_bbox)
  pred_boxes = np.array(pred_bbox)

  true_positives = 0
  false_positives = 0
  false_negatives = 0

  for pred_box in pred_boxes:
    iou = calculate_iou(gt_bbox, pred_box)
    if np.max(iou) >= iou_threshold:
        true_positives += 1
    else:
        false_positives += 1

  false_negatives = len(gt_bbox) - true_positives

  precision = true_positives / (true_positives + false_positives)
  recall = true_positives / (true_positives + false_negatives)

  if precision == 0 or recall == 0:
    f1_score = 0
  else:
    f1_score = 2 * (precision * recall) / (precision + recall)

  return precision, recall, f1_score

def calculation_accuracy(detection_per_frame, ground_truth_per_frame):
   """
   param: detection_per_frame - list of integers representing number of human detections per frame.
   param: ground_truth_per_frame - list of integers representing number of actual humans in the frame.
   """
   if len(detection_per_frame) != len(ground_truth_per_frame):
      raise ValueError("Different number of detections and ground truths")
   
   correct_detections = sum(1 for detected, actual in zip(detection_per_frame, ground_truth_per_frame) if detected == actual)
   accuracy = correct_detections / len(ground_truth_per_frame) 
   return f'{accuracy*100:.2f}%'

def print_evaluation_table(detection_methods, results, results2):
    """
    Print a table with precision, recall, and F1 scores for multiple detection methods.

    Parameters:
    - detection_methods: List of names of detection methods.
    - results: Dictionary containing evaluation results for each method.
    """
    df = pd.DataFrame(results, index=['Precision', 'Recall', 'F1 Score'])
    df.columns.name = 'Detection Method'
    accuracy_df = pd.DataFrame(results2, index=['Human Detection Accuracy'])
    df = pd.concat([df, accuracy_df])
    print(df)

#Example usage. It supposes that you have list of bounding boxes coordinates both predicted and ground truth one.
gt_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
pred_boxes_method1 = [[90, 90, 210, 210], [320, 320, 420, 420], [500, 500, 600, 600]]
pred_boxes_method2 = [[100, 100, 200, 200], [310, 310, 410, 410], [520, 520, 620, 620]]
iou_threshold = 0.5

# Example usage (assuming you have these lists from your video processing)
yolo_detections = [3, 2, 4, 3, 5]  # Example detection counts per frame from YOLO
hog_detections = [2, 2, 4, 3, 5]   # Example detection counts per frame from HOG
ground_truth = [3, 2, 4, 3, 5]     # Actual counts of humans per frame

results = {}
results['YOLOv4'] = evaluate_model(gt_boxes, pred_boxes_method1, iou_threshold)
results['HOG'] = evaluate_model(gt_boxes, pred_boxes_method2, iou_threshold)
# print_evaluation_table(['YOLOv4', 'HOG'], results)

results2={}
results2['YOLOv4'] = calculation_accuracy(yolo_detections, ground_truth)
results2['HOG'] = calculation_accuracy(hog_detections, ground_truth)
print_evaluation_table(['YOLOv4', 'HOG'], results, results2)




  
  
    



