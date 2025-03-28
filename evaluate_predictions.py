import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
from loguru import logger
from tqdm import tqdm

def compute_iou(box1, box2):
    """
    计算两个边界框的三维 Intersection over Union (IoU)。
    边界框格式: [x1, y1, x2, y2, z1, z2]
    """
    # 计算交集
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    z_front = max(box1[4], box2[4])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    z_back = min(box1[5], box2[5])

    if x_right < x_left or y_bottom < y_top or z_back < z_front:
        return 0.0

    intersection_volume = (x_right - x_left) * (y_bottom - y_top) * (z_back - z_front)

    # 计算各自的体积
    box1_volume = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    box2_volume = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])

    union_volume = box1_volume + box2_volume - intersection_volume

    if union_volume == 0:
        return 0.0

    return intersection_volume / union_volume

def load_ground_truth_boxes(gt_npz_path):
    """
    加载地面真实边界框和类别数据。

    参数:
        gt_npz_path (Path): 地面真实 .npz 文件的路径。

    返回:
        tuple:
            boxes_gt (ndarray): 地面真实边界框数组，形状为 (N, 6)。
            classes_gt (ndarray): 地面真实类别数组，形状为 (N,)。
    """
    try:
        data = np.load(gt_npz_path, allow_pickle=True)
        if 'boxes' in data and 'classes' in data:
            boxes_gt = data['boxes']
            classes_gt = data['classes']
            logger.debug(f"加载 {gt_npz_path.name} 成功，包含 {len(boxes_gt)} 个边界框和对应的类别。")
            return boxes_gt, classes_gt
        else:
            logger.warning(f"'boxes' 或 'classes' 键不存在于 {gt_npz_path.name} 中。")
            return None, None
    except Exception as e:
        logger.warning(f"无法加载地面真实边界框数据于 {gt_npz_path.name}: {e}. Skipping.")
        return None, None

def load_pickle_file(pkl_path):
    """
    加载预测的 pickle 文件。

    参数:
        pkl_path (Path): 预测的 pickle 文件路径。

    返回:
        dict: 包含 'pred_boxes', 'pred_scores', 'pred_labels' 的字典。
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            if all(k in data for k in ('pred_boxes', 'pred_scores', 'pred_labels')):
                logger.debug(f"加载 {pkl_path.name} 成功，包含 {len(data['pred_boxes'])} 个预测边界框。")
                return data
            else:
                logger.warning(f"{pkl_path.name} 缺少必要的键 ('pred_boxes', 'pred_scores', 'pred_labels')。")
                return None
    except Exception as e:
        logger.warning(f"无法加载预测文件 {pkl_path.name}: {e}. Skipping.")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Process nnDetection predictions, compute AUC, Sensitivity, and Specificity for multi-class classification.')
    parser.add_argument("--pkl_dir", type=str, required=True,
                        help="Path to the directory containing model prediction files (nnDetection bounding boxes)")
    parser.add_argument("--gt_npz_dir", type=str, required=True,
                        help="Path to the directory containing ground truth bounding box files (_boxes_gt.npz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output results (metrics)")
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="Confidence threshold for predictions. Default: 0.2")
<<<<<<< HEAD
    parser.add_argument("--max_predictions", type=int, default=3,
                        help="Maximum number of predictions to keep per case after filtering. Default: 3")
=======
    parser.add_argument("--max_predictions", type=int, default=2,
                        help="Maximum number of predictions to keep per case after filtering. Default: 2")
>>>>>>> 864d4cade90dccff407e62481e9e8e0b38c746b0
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold to consider a prediction as True Positive. Default: 0.5")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level. Default: INFO")
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置日志级别
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    # 确保输出目录存在
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有预测文件 ID（假设文件名格式为 {case_id}_boxes.pkl）
    pkl_files = list(Path(args.pkl_dir).glob("*_boxes.pkl"))
    case_ids = [p.stem.replace('_boxes', '') for p in pkl_files]

    if not case_ids:
        logger.error("No *_boxes.pkl files found in the specified pkl_dir.")
        sys.exit(1)
    else:
        logger.info(f"Found {len(case_ids)} cases.")

    # 获取所有地面真实文件
    gt_npz_files = list(Path(args.gt_npz_dir).glob("*_boxes_gt.npz"))
    gt_case_ids = [p.stem.replace('_boxes_gt', '') for p in gt_npz_files]

    # 确保每个预测文件都有对应的地面真实文件
    case_ids = [cid for cid in case_ids if cid in gt_case_ids]
    if not case_ids:
        logger.error("No matching *_boxes_gt.npz files found for the prediction files.")
        sys.exit(1)
    else:
        logger.info(f"{len(case_ids)} cases have matching ground truth files.")

    # 初始化指标列表
    all_true_labels = []
    all_pred_scores = []
    all_pred_labels = []

    for cid in tqdm(case_ids, desc="Processing cases"):
        # 加载预测边界框
        pkl_path = Path(args.pkl_dir) / f"{cid}_boxes.pkl"
        pred_data = load_pickle_file(pkl_path)
        if pred_data is None:
            continue

        pred_boxes = pred_data.get("pred_boxes", [])
        pred_scores = pred_data.get("pred_scores", [])
        pred_labels = pred_data.get("pred_labels", [])

        if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
            logger.warning(f"Inconsistent lengths in predictions for PID: {cid}. Skipping.")
            continue

        # 筛选置信度高于阈值的预测框
        filtered_pred = []
        filtered_scores = []
        filtered_labels = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score < args.confidence_threshold:
                continue
            if label < 0:
                continue  # 排除背景或无效标签
            try:
                label = int(label)
            except ValueError:
                logger.warning(f"Label值不能转换为整数: {label}. Skipping.")
                continue
            filtered_pred.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)

        if not filtered_pred:
            logger.warning(f"No predictions above confidence threshold for PID: {cid}. Skipping.")
            continue

        # 仅保留最高得分的前 max_predictions 个预测
        sorted_indices = np.argsort(filtered_scores)[::-1][:args.max_predictions]
        filtered_pred = [filtered_pred[i] for i in sorted_indices]
        filtered_scores = [filtered_scores[i] for i in sorted_indices]
        filtered_labels = [filtered_labels[i] for i in sorted_indices]

        # 加载地面真实边界框和类别
        gt_npz_path = Path(args.gt_npz_dir) / f"{cid}_boxes_gt.npz"
        boxes_gt, classes_gt = load_ground_truth_boxes(gt_npz_path)
        if boxes_gt is None or classes_gt is None:
            logger.warning(f"地面真实数据无效于 PID: {cid}. Skipping.")
            continue

        # 进行匹配并收集标签和得分
        for gt_box, gt_class in zip(boxes_gt, classes_gt):
            matched = False
            best_iou = 0.0
            best_pred_score = 0.0
            for pred_box, pred_score, pred_label in zip(filtered_pred, filtered_scores, filtered_labels):
                iou = compute_iou(pred_box, gt_box)
                if iou >= args.iou_threshold and pred_label == gt_class:
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_score = pred_score
                        matched = True
            all_true_labels.append(gt_class)
            all_pred_scores.append(best_pred_score if matched else 0.0)
            all_pred_labels.append(gt_class if matched else 0)

    if not all_true_labels:
        logger.error("No valid matches found. Exiting.")
        sys.exit(1)

    # 计算 AUC for each class
    unique_classes = sorted(list(set(all_true_labels)))
    auc_scores = {}
    for cls in unique_classes:
        binary_true = [1 if label == cls else 0 for label in all_true_labels]
        binary_scores = [score if label == cls else 0.0 for label, score in zip(all_true_labels, all_pred_scores)]
        try:
            auc_score = roc_auc_score(binary_true, binary_scores)
            auc_scores[cls] = auc_score
        except ValueError:
            auc_scores[cls] = np.nan
            logger.warning(f"无法计算 AUC for class {cls}，可能因为所有 y_true 都是同一个类别。")

    # 生成分类报告
    report = classification_report(all_true_labels, all_pred_labels, labels=unique_classes, digits=4)
    report_path = Path(args.output_dir) / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")

    # 保存 AUC scores
    auc_df = pd.DataFrame({
        'Class': unique_classes,
        'AUC': [auc_scores[cls] for cls in unique_classes]
    })
    auc_df.to_csv(Path(args.output_dir) / "auc_scores.csv", index=False)
    logger.info(f"AUC scores saved to {Path(args.output_dir) / 'auc_scores.csv'}")

    # 计算并保存混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{cls}" for cls in unique_classes],
                         columns=[f"Pred_{cls}" for cls in unique_classes])
    cm_path = Path(args.output_dir) / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # 计算敏感度（召回率）和特异性
    sensitivity = {}
    specificity = {}
    for idx, cls in enumerate(unique_classes):
        TP = cm[idx, idx]
        FN = cm[idx, :].sum() - TP
        FP = cm[:, idx].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        sensitivity[cls] = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
        specificity[cls] = TN / (TN + FP) if (TN + FP) > 0 else float('nan')

    # 保存敏感度和特异性
    metrics_df = pd.DataFrame({
        'Class': unique_classes,
        'Sensitivity (Recall)': [sensitivity[cls] for cls in unique_classes],
        'Specificity': [specificity[cls] for cls in unique_classes],
        'AUC': [auc_scores[cls] for cls in unique_classes]
    })
    metrics_df.to_csv(Path(args.output_dir) / "metrics.csv", index=False)
    logger.info(f"Sensitivity, Specificity, and AUC saved to {Path(args.output_dir) / 'metrics.csv'}")

if __name__ == "__main__":
    main()