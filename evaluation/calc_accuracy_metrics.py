import os
import numpy as np
import glob

GT_PATH = "/home/hongik/Desktop/mAP_1218_TJ/archive/css-data/valid/labels"     
PRED_PATH = "/home/hongik/Desktop/512_folder/prediction_350.1_TJ"     


TARGET_CLASSES = [0, 5, 7]
CLASS_NAMES = {0: 'Hardhat', 5: 'Person', 7: 'Safety Vest'}


CONF_THRESHOLD = 0.5


def parse_txt(file_path):
    
    boxes = []
    if not os.path.exists(file_path):
        return np.array([])
   
    with open(file_path, 'r') as f:
        lines = f.readlines()
       
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])
        x, y, w, h = parts[1], parts[2], parts[3], parts[4]

        score = parts[5] if len(parts) > 5 else 1.0
        boxes.append([cls, x, y, w, h, score])
       
    return np.array(boxes)

def compute_iou(box1, box2):
    
    b1_x1, b1_x2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
    b1_y1, b1_y2 = box1[2] - box1[4]/2, box1[2] + box1[4]/2
   
    b2_x1, b2_x2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2
    b2_y1, b2_y2 = box2[2] - box2[4]/2, box2[2] + box2[4]/2
   
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
   
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
   
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
   
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(recalls, precisions):
    
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
   
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
       
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def calculate_map():
    print(f"[평가 시작] GT: {GT_PATH}")
    print(f"[평가 대상] PRED: {PRED_PATH}")
    print(f"[설정] Confidence Threshold: {CONF_THRESHOLD}")
   
    pred_files = glob.glob(os.path.join(PRED_PATH, "*.txt"))
    if len(pred_files) == 0:
        print("경고: 추론 결과 파일(.txt)을 찾을 수 없습니다.")
        return

    aps = []
   
   
    print("\n" + "="*85)
    print(f"{'Class':<12} {'AP@0.5':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'GT Count':<10}")
    print("="*85)

    for c in TARGET_CLASSES:
        true_positives = []
        scores = []
        n_gt = 0
       
       
        for pred_file in pred_files:
            file_id = os.path.basename(pred_file)
            gt_file = os.path.join(GT_PATH, file_id)
           
            pred_boxes = parse_txt(pred_file)
            gt_boxes = parse_txt(gt_file)
           
           
            if len(pred_boxes) > 0:
                pred_c = pred_boxes[pred_boxes[:, 0] == c]
            else:
                pred_c = np.array([])
               
            if len(gt_boxes) > 0:
                gt_c = gt_boxes[gt_boxes[:, 0] == c]
            else:
                gt_c = np.array([])
           
            n_gt += len(gt_c)
           
            if len(pred_c) == 0 and len(gt_c) == 0:
                continue
           
           
            if len(pred_c) == 0:
                continue
           
            
            if len(gt_c) == 0:
                for _ in pred_c:
                    scores.append(_[5])
                    true_positives.append(0)
                continue
           
            
            pred_c = pred_c[(-pred_c[:, 5]).argsort()]
            gt_detected = [False] * len(gt_c)
           
            for p_box in pred_c:
                scores.append(p_box[5])
               
                best_iou = 0
                best_gt_idx = -1
               
                
                for i, g_box in enumerate(gt_c):
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
               
                
                if best_iou >= 0.5:
                    if not gt_detected[best_gt_idx]:
                        true_positives.append(1) # 정답 (TP)
                        gt_detected[best_gt_idx] = True
                    else:
                        true_positives.append(0) 
                else:
                    true_positives.append(0) 
                   
        
        if n_gt == 0:
            print(f"{CLASS_NAMES[c]:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {0:<10}")
            continue
           
        scores = np.array(scores)
        true_positives = np.array(true_positives)
       
        
        indices = np.argsort(-scores)
        tp_sorted = true_positives[indices]
       
        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(1 - tp_sorted)
       
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

        
       
        valid_indices = scores >= CONF_THRESHOLD
       
        
        tp_count = np.sum(true_positives[valid_indices])
       
        
        fp_count = len(true_positives[valid_indices]) - tp_count
       
        
        fn_count = n_gt - tp_count
       
        epsilon = 1e-7
        final_precision = tp_count / (tp_count + fp_count + epsilon)
        final_recall = tp_count / (tp_count + fn_count + epsilon)
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + epsilon)
       
        print(f"{CLASS_NAMES[c]:<12} {ap:.4f}     {final_precision:.4f}       {final_recall:.4f}     {final_f1:.4f}     {n_gt:<10}")

    print("="*85)
    if len(aps) > 0:
        print(f"\n✅ 최종 mAP@0.5: {np.mean(aps):.4f}")
    else:
        print("\n평가할 데이터가 없습니다.")

if __name__ == "__main__":
    calculate_map()
