# def calculate_model_accuracy(detections_per_frame, ground_truth_per_frame):
#     """
#     Calculate model accuracy based on frame-by-frame comparison with ground truth.

#     :param detections_per_frame: List of integers, number of detections per frame by the model.
#     :param ground_truth_per_frame: List of integers, actual number of objects per frame.
#     :return: Accuracy as a float.
#     """
#     if len(detections_per_frame) != len(ground_truth_per_frame):
#         print("Error: Mismatch in number of frames between detections and ground truth.")
#         return
    
#     # Calculate accuracy per frame
#     correct_detections = sum(1 for detected, actual in zip(detections_per_frame, ground_truth_per_frame) if detected == actual)
#     accuracy = correct_detections / len(ground_truth_per_frame)
    
#     print(f"Model Accuracy: {accuracy*100:.2f}%")
#     return accuracy

# # Example usage (assuming you have these lists from your video processing)
# yolo_detections = [3, 2, 4, 3, 5]  # Example detection counts per frame from YOLO
# hog_detections = [2, 2, 4, 3, 5]   # Example detection counts per frame from HOG
# ground_truth = [3, 2, 4, 3, 5]     # Actual counts of humans per frame

# # Calculate and print the accuracy for both models
# yolo_accuracy = calculate_model_accuracy(yolo_detections, ground_truth)
# hog_accuracy = calculate_model_accuracy(hog_detections, ground_truth)
import yolov4, HOG
import cv2
import numpy as np
def evaluate_model_accuracy(video_path, ground_truth_data, detection_functions):
    """
    Compare model accuracies using precision, recall, and F1-score.

    Args:
    - video_path: path to the video file.
    - ground_truth_data: a dictionary where keys are frame indices and values are lists of bounding boxes (x, y, w, h) for that frame.
    - detection_functions: a list of tuples containing the detection function and its name.

    Returns:
    - A dictionary containing precision, recall, and F1-scores for each model.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_index = 0
    results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For each detection model, run detection and compute accuracy
        for detect_func, name in detection_functions:
            detected_boxes = detect_func(frame)
            gt_boxes = ground_truth_data.get(frame_index, [])

            # Compute matches based on IoU, precision, recall, and F1-score
            tp, fp, fn = 0, 0, 0
            for db in detected_boxes:
                match_found = False
                for gt in gt_boxes:
                    if iou(db, gt) >= 0.5:  # assuming IoU threshold of 0.5
                        tp += 1
                        match_found = True
                        break
                if not match_found:
                    fp += 1
            fn = len(gt_boxes) - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            # Store results
            if name not in results:
                results[name] = {'precision': [], 'recall': [], 'f1_score': []}
            results[name]['precision'].append(precision)
            results[name]['recall'].append(recall)
            results[name]['f1_score'].append(f1_score)

        frame_index += 1

    # Release video and resources
    cap.release()

    # Average the results across all frames
    for model in results:
        for metric in results[model]:
            results[model][metric] = np.mean(results[model][metric])

    return results

# Helper function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    wi = max(0, xi2 - xi1)
    hi = max(0, yi2 - yi1)
    area_intersection = wi * hi

    # Calculate union
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_union = area_box1 + area_box2 - area_intersection

    return area_intersection / area_union if area_union != 0 else 0

# Example usage
video_path = "C:\\Users\\TheEarthG\\Downloads\\human.mp4"
ground_truth_data = {0: [(10, 20, 50, 60)], 1: [(15, 25, 55, 65)], 2: [(20, 30, 50, 60)]}
detection_functions = [
    (lambda frame: yolov4.detect_objects(frame, "C:\\Users\\TheEarthG\\Downloads\\yolov4.weights", "C:\\Users\\TheEarthG\\Downloads\\yolov4.cfg", "C:\\Users\\TheEarthG\\Downloads\\coco.names"), "YOLOv4"),
    (lambda frame: HOG.detect_people_with_hog(frame), "HOG_SVM")
]

results = evaluate_model_accuracy(video_path, ground_truth_data, detection_functions)
print(results)

