import cv2
import numpy as np

# Function to calculate the Intersection Over Union (IOU)
def calculate_iou(boxA, boxB):
    # Calculate intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Calculate union
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    # Return IOU
    iou = inter_area / union_area
    return iou

# Function to calculate precision, recall, and F1-score
def evaluate_detections(ground_truths, detected_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # For each ground truth, check if it is matched with a detection
    for gt_box in ground_truths:
        matched = False
        for det_box in detected_boxes:
            iou = calculate_iou(gt_box, det_box)
            if iou >= iou_threshold:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_negatives += 1

    # False positives are detections without a matching ground truth
    false_positives = len(detected_boxes) - true_positives

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Define ground truth bounding boxes
# Format: [x, y, width, height]
ground_truths = [
    [10, 20, 30, 40],  # Example ground truth box
    # Add more ground truth boxes as needed
]

# Define detected bounding boxes for YOLOv4
detected_boxes_yolov4 = [
    [10, 18, 26, 42],  # Example detected box
    # Add more detected boxes as needed
]

# Define detected bounding boxes for HOG
detected_boxes_hog = [
    [15, 25, 30, 40],  # Example detected box
    # Add more detected boxes as needed
]

# Calculate metrics for YOLOv4
precision_yolov4, recall_yolov4, f1_score_yolov4 = evaluate_detections(ground_truths, detected_boxes_yolov4)

# Calculate metrics for HOG
precision_hog, recall_hog, f1_score_hog = evaluate_detections(ground_truths, detected_boxes_hog)

# Print results
print("YOLOv4 - Precision:", precision_yolov4, "Recall:", recall_yolov4, "F1-Score:", f1_score_yolov4)
print("HOG - Precision:", precision_hog, "Recall:", recall_hog, "F1-Score:", f1_score_hog)
