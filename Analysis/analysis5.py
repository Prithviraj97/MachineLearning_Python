import tensorflow as tf
def evaluate_object_detection(ground_truth_boxes, detected_boxes, detection_methods):
  """
  Evaluates object detection performance for multiple methods.

  Args:
      ground_truth_boxes: List of ground truth bounding boxes for each image.
                          Each box is a list of [y_min, x_min, y_max, x_max] coordinates.
      detected_boxes: List of detected bounding boxes for each image from different methods.
                       Each element is a dictionary with keys as detection methods (strings)
                       and values as lists of detections. Each detection is a list of
                       [y_min, x_min, y_max, x_max, confidence_score] coordinates.
      detection_methods: List of detection method names (strings).

  Returns:
      A dictionary containing precision, recall, and F1-score for each detection method.
  """

  # Initialize empty dictionaries to store evaluation metrics
  precisions = {method: [] for method in detection_methods}
  recalls = {method: [] for method in detection_methods}
  f1_scores = {method: [] for method in detection_methods}

  # Loop through each image
  for gt_boxes, det_boxes_dict in zip(ground_truth_boxes, detected_boxes):
    # Loop through each detection method
    # for method, detections in det_boxes_dict.items():
    for method in detection_methods:
      detections = det_boxes_dict[method]
      # Calculate true positives, false positives, and false negatives
      true_positives = 0
      false_positives = 0
      false_negatives = len(gt_boxes)

      # Loop through each ground truth box
      for gt_box in gt_boxes:
        # Check for overlaps with detected boxes for this method
        iou_max = 0
        for det_box in detections:
          iou = calculate_iou(gt_box, det_box)  # Replace with your IOU calculation function
          iou_max = max(iou_max, iou)

        # Consider a ground truth box detected if IoU is above a threshold (e.g., 0.5)
        if iou_max >= 0.5:
          true_positives += 1
          false_negatives -= 1

      # Calculate false positives from detections without matching ground truth boxes
      false_positives = len(detections) - true_positives

      # Calculate precision, recall, and F1-score
      if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
      else:
        precision = 0.0
      if len(gt_boxes) > 0:
        recall = true_positives / len(gt_boxes)
      else:
        recall = 0.0
      if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
      else:
        f1_score = 0.0

      # Append metrics for this method and image
      precisions[method].append(precision)
      recalls[method].append(recall)
      f1_scores[method].append(f1_score)

  # Calculate average precision, recall, and F1-score for each method
  average_metrics = {}
  for method in detection_methods:
    average_metrics[method] = {
        "precision": tf.reduce_mean(precisions[method]),
        "recall": tf.reduce_mean(recalls[method]),
        "f1_score": tf.reduce_mean(f1_scores[method]),
    }

  return average_metrics

def calculate_iou(gt_box, det_box):
  """
  Calculates Intersection-over-Union (IoU) between two bounding boxes.

  Args:
      gt_box: Ground truth bounding box (list of [y_min, x_min, y_max, x_max] coordinates).
      det_box: Detected bounding box (list of [y_min, x_min, y_max, x_max, confidence_score] coordinates).

  Returns:
      IoU value (float) between 0 and 1.
  """
  # Calculate area of overlap and area of union
  ymin_intersection = max(gt_box[0], det_box[0])
  xmin_intersection = max(gt_box[1], det_box[1])
  ymax_intersection = min(gt_box[2], det_box[2])
  xmax_intersection = min(gt_box[3], det_box[3])

  area_intersection = max(0, xmax_intersection - xmin_intersection) * max(0, ymax_intersection - ymin_intersection)
  area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
  area_det = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
  area_union = area_gt + area_det - area_intersection

  # Calculate IoU
  iou = area_intersection / (area_union + 1e-10)  # Add a small value to avoid division by zero
  return iou

# Example usage
ground_truth_boxes = [
    [0.2, 0.3, 0.7, 0.8],  # Example ground truth bounding box
    # ... more ground truth boxes
]

detected_boxes = {
    "method_1": [
        [0.1, 0.2, 0.8, 0.9, 0.9],  # Example detection from method 1
        # ... more detections for method 1
    ],
    "method_2": [
        # ... detections from method 2
    ],
    # ... detections from other methods
}

detection_methods = ["method_1", "method_2", ...]  # List of detection method names

evaluation_results = evaluate_object_detection(ground_truth_boxes, detected_boxes, detection_methods)

# Print evaluation results for each method
for method, metrics in evaluation_results.items():
  print(f"Method: {method}")
  print(f"  Precision: {metrics['precision']}")
  print(f"  Recall: {metrics['recall']}")
  print(f"  F1-Score: {metrics['f1_score']}")  

