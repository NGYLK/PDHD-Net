import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm
from scipy import integrate
from sklearn.metrics import roc_auc_score

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

class FROCAnalyzer:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        # 定义病灶类型
        self.clinically_significant_threshold = 2  # 类别≥2为临床显著
        
    def load_data(self, gt_dir, pred_dir, confidence_threshold=0.2):
        """加载并预处理数据"""
        case_results = {}
        
        # 获取所有预测文件和真实标注文件
        pkl_files = list(Path(pred_dir).glob("*_boxes.pkl"))
        case_ids = [p.stem.replace('_boxes', '') for p in pkl_files]
        
        # 获取所有真实标注文件
        gt_npz_files = list(Path(gt_dir).glob("*_boxes_gt.npz"))
        gt_case_ids = [p.stem.replace('_boxes_gt', '') for p in gt_npz_files]
        
        # 确保每个预测文件都有对应的真实标注文件
        case_ids = [cid for cid in case_ids if cid in gt_case_ids]
        logger.info(f"Found {len(case_ids)} cases with matching ground truth files")
        
        for cid in tqdm(case_ids, desc="Loading data"):
            # 加载预测数据
            pkl_path = Path(pred_dir) / f"{cid}_boxes.pkl"
            pred_data = load_pickle_file(pkl_path)
            if pred_data is None:
                continue
                
            pred_boxes = pred_data.get("pred_boxes", [])
            pred_scores = pred_data.get("pred_scores", [])
            pred_labels = pred_data.get("pred_labels", [])
            
            if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
                logger.warning(f"Inconsistent lengths in predictions for PID: {cid}. Skipping.")
                continue
                
            # 加载真实标注
            gt_npz_path = Path(gt_dir) / f"{cid}_boxes_gt.npz"
            boxes_gt, classes_gt = load_ground_truth_boxes(gt_npz_path)
            if boxes_gt is None or classes_gt is None:
                logger.warning(f"Invalid ground truth data for PID: {cid}. Skipping.")
                continue
                
            # 筛选置信度高于阈值的预测
            filtered_pred = []
            filtered_scores = []
            filtered_labels = []
            
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < confidence_threshold:
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
                
            case_results[cid] = {
                'gt_boxes': boxes_gt,
                'gt_classes': classes_gt,
                'pred_boxes': filtered_pred,
                'pred_scores': filtered_scores,
                'pred_labels': filtered_labels
            }
            
        return case_results
            
    def evaluate_froc(self, case_results):
        """计算FROC曲线和性能指标"""
        # 用于存储评估结果的数据结构
        results = {
            'index': {'fps_list': [], 'sens_list': [], 'thresholds': []},
            'clinical': {'fps_list': [], 'sens_list': [], 'thresholds': []},
            'all': {'fps_list': [], 'sens_list': [], 'thresholds': []}
        }
        
        # 阈值范围
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        # 对每个阈值进行评估
        for threshold in thresholds:
            # 评估三种类型的病灶
            for lesion_type in ['index', 'clinical', 'all']:
                fps, sens = self._evaluate_at_threshold(case_results, threshold, lesion_type)
                results[lesion_type]['fps_list'].append(fps)
                results[lesion_type]['sens_list'].append(sens)
                results[lesion_type]['thresholds'].append(threshold)
                
        # 计算特定指标
        metrics = {}
        
        # 确保计算所有需要报告的指标
        for lesion_type in ['index', 'clinical', 'all']:
            fps = np.array(results[lesion_type]['fps_list'])
            sens = np.array(results[lesion_type]['sens_list'])
            
            # 确保数据按敏感度排序
            sort_idx = np.argsort(sens)
            fps_sorted = fps[sort_idx]
            sens_sorted = sens[sort_idx]
            
            # 计算特定敏感度下的FP率
            metrics[f"{lesion_type}_FP@Sen80%"] = self._interpolate_at_sensitivity(fps_sorted, sens_sorted, 0.8)
            if lesion_type in ['index', 'clinical']:
                metrics[f"{lesion_type}_FP@Sen90%"] = self._interpolate_at_sensitivity(fps_sorted, sens_sorted, 0.9)
            if lesion_type == 'all':
                metrics[f"{lesion_type}_FP@Sen60%"] = self._interpolate_at_sensitivity(fps_sorted, sens_sorted, 0.6)
            
            # 计算特定FP率下的敏感度
            metrics[f"{lesion_type}_Sen@FP1"] = self._interpolate_at_fp(fps, sens, 1.0)
            
            # 计算部分AUC
            metrics[f"{lesion_type}_pAUC(0.01-1)"] = self._compute_partial_auc(fps, sens, 0.01, 1.0)
            metrics[f"{lesion_type}_pAUC(0.1-3)"] = self._compute_partial_auc(fps, sens, 0.1, 3.0)
        
        # 计算不同Gleason评分组的性能
        gs_metrics = self._evaluate_by_gleason_score(case_results)
        metrics.update(gs_metrics)
        
        return results, metrics, case_results  # 返回case_results供后续使用
    
    def _evaluate_at_threshold(self, case_results, threshold, lesion_type):
        """在特定阈值和病灶类型下评估性能"""
        total_gt = 0  # 真实病灶总数
        total_tp = 0  # 真阳性总数
        total_fp = 0  # 假阳性总数
        n_patients = len(case_results)
        
        for case_id, case_data in case_results.items():
            # 根据病灶类型筛选真实病灶
            gt_boxes, gt_classes = self._filter_lesions_by_type(
                case_data['gt_boxes'],
                case_data['gt_classes'],
                lesion_type
            )
            
            # 如果没有符合条件的真实病灶，跳过
            if len(gt_boxes) == 0:
                continue
                
            total_gt += len(gt_boxes)
            
            # 筛选高于阈值的预测
            pred_mask = np.array(case_data['pred_scores']) >= threshold
            if not any(pred_mask):
                continue
                
            pred_boxes = np.array(case_data['pred_boxes'])[pred_mask]
            pred_scores = np.array(case_data['pred_scores'])[pred_mask]
            pred_labels = np.array(case_data['pred_labels'])[pred_mask]
            
            # 标记真实病灶是否被检测到
            gt_detected = [False] * len(gt_boxes)
            
            # 记录已匹配的预测框
            pred_matched = [False] * len(pred_boxes)
            
            # 对每个预测框，检查是否与真实病灶匹配
            for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    # 检查IoU和类别是否匹配
                    iou = compute_iou(pred_box, gt_box)
                    
                    if iou >= self.iou_threshold and not gt_detected[gt_idx] and not pred_matched[pred_idx]:
                        # 检测到真实病灶
                        gt_detected[gt_idx] = True
                        pred_matched[pred_idx] = True
                        total_tp += 1
                        break
            
            # 计算假阳性
            total_fp += sum(1 for matched in pred_matched if not matched)
        
        # 计算敏感度和每患者假阳性率
        sensitivity = total_tp / total_gt if total_gt > 0 else 0
        fp_per_patient = total_fp / n_patients if n_patients > 0 else 0
        
        return fp_per_patient, sensitivity
    
    def _filter_lesions_by_type(self, boxes, classes, lesion_type):
        """根据病灶类型筛选病灶"""
        if lesion_type == 'index':
            mask = np.array(classes) == 4  # 只选择类别4的病变
            return np.array(boxes)[mask], np.array(classes)[mask]
        elif lesion_type == 'clinical':
            # 筛选临床显著病变 (类别 >= 2)
            mask = np.array(classes) >= self.clinically_significant_threshold
            return np.array(boxes)[mask], np.array(classes)[mask]
        else:  # 'all'
            # 包括所有病灶
            return boxes, classes
    
    def _interpolate_at_sensitivity(self, fps, sensitivities, target_sensitivity):
        """在特定敏感度处插值计算FP率"""
        # 排序以确保插值正确
        idx = np.searchsorted(sensitivities, target_sensitivity)
        
        if idx == 0:
            return fps[0]
        elif idx == len(sensitivities):
            return fps[-1]
            
        # 线性插值
        s0, s1 = sensitivities[idx-1], sensitivities[idx]
        f0, f1 = fps[idx-1], fps[idx]
        
        # 计算插值
        if s1 == s0:
            return f0
        else:
            return f0 + (f1 - f0) * (target_sensitivity - s0) / (s1 - s0)
    
    def _interpolate_at_fp(self, fps, sensitivities, target_fp):
        """在特定FP率处插值计算敏感度"""
        # 确保数据按FP排序
        sort_idx = np.argsort(fps)
        sorted_fps = fps[sort_idx]
        sorted_sens = sensitivities[sort_idx]
        
        idx = np.searchsorted(sorted_fps, target_fp)
        
        if idx == 0:
            return sorted_sens[0]
        elif idx == len(sorted_fps):
            return sorted_sens[-1]
            
        # 线性插值
        f0, f1 = sorted_fps[idx-1], sorted_fps[idx]
        s0, s1 = sorted_sens[idx-1], sorted_sens[idx]
        
        # 计算插值
        if f1 == f0:
            return s0
        else:
            return s0 + (s1 - s0) * (target_fp - f0) / (f1 - f0)
    
    def _compute_partial_auc(self, fps, sensitivities, min_fp, max_fp):
        """Calculate partial AUC"""
        # Ensure data is sorted by FP rate
        sort_idx = np.argsort(fps)
        sorted_fps = fps[sort_idx]
        sorted_sens = sensitivities[sort_idx]
        
        # Filter data points within specified FP range
        mask = (sorted_fps >= min_fp) & (sorted_fps <= max_fp)
        range_fps = sorted_fps[mask]
        range_sens = sorted_sens[mask]
        
        if len(range_fps) < 2:
            return 0.0
        
        # Use numpy's trapz instead of scipy's (which is deprecated)
        return np.trapz(range_sens, range_fps)
    
    def _evaluate_by_gleason_score(self, case_results):
        """Evaluate performance by Gleason score groups"""
        # Define Gleason score groups
        gs_groups = {
            'GGG1': [0],  # Assuming class 0 corresponds to GGG1
            'GGG2': [1],  # Assuming class 1 corresponds to GGG2
            'GGG3': [2],  # Assuming class 2 corresponds to GGG3
            'GGG4': [3],  # Assuming class 3 corresponds to GGG4
            'GGG5': [4]   # Assuming class 4 corresponds to GGG5
        }
        
        metrics = {}
        
        # Calculate performance for each Gleason score group
        for gs_name, gs_classes in gs_groups.items():
            metrics[f"{gs_name}_FP@Sen60%"] = self._compute_fp_at_sensitivity(case_results, gs_classes, 0.6)
            metrics[f"{gs_name}_FP@Sen80%"] = self._compute_fp_at_sensitivity(case_results, gs_classes, 0.8)
            metrics[f"{gs_name}_FP@Sen90%"] = self._compute_fp_at_sensitivity(case_results, gs_classes, 0.9)
            
        return metrics
    
    def _compute_fp_at_sensitivity(self, case_results, gs_classes, target_sensitivity):
        """计算特定Gleason评分组在给定敏感度下的FP率"""
        # 阈值范围
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        fps_list = []
        sens_list = []
        
        for threshold in thresholds:
            fps, sens = self._evaluate_gs_at_threshold(case_results, gs_classes, threshold)
            fps_list.append(fps)
            sens_list.append(sens)
            
        # 插值计算特定敏感度下的FP率
        sort_idx = np.argsort(sens_list)
        sorted_fps = np.array(fps_list)[sort_idx]
        sorted_sens = np.array(sens_list)[sort_idx]
        
        return self._interpolate_at_sensitivity(sorted_fps, sorted_sens, target_sensitivity)
    
    def _evaluate_gs_at_threshold(self, case_results, gs_classes, threshold):
        """在特定阈值下评估特定Gleason评分组的性能"""
        total_gt = 0
        total_tp = 0
        total_fp = 0
        n_patients = len(case_results)
        
        for case_id, case_data in case_results.items():
            # 筛选指定Gleason评分的真实病灶
            gs_mask = np.isin(case_data['gt_classes'], gs_classes)
            if not any(gs_mask):
                continue
                
            gs_boxes = np.array(case_data['gt_boxes'])[gs_mask]
            gs_classes = np.array(case_data['gt_classes'])[gs_mask]
            
            total_gt += len(gs_boxes)
            
            # 筛选高于阈值的预测
            pred_mask = np.array(case_data['pred_scores']) >= threshold
            if not any(pred_mask):
                continue
                
            pred_boxes = np.array(case_data['pred_boxes'])[pred_mask]
            pred_scores = np.array(case_data['pred_scores'])[pred_mask]
            pred_labels = np.array(case_data['pred_labels'])[pred_mask]
            
            # 标记真实病灶是否被检测到
            gt_detected = [False] * len(gs_boxes)
            
            # 记录已匹配的预测框
            pred_matched = [False] * len(pred_boxes)
            
            # 对每个预测框，检查是否与真实病灶匹配
            for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gs_boxes, gs_classes)):
                    # 检查IoU和类别是否匹配
                    iou = compute_iou(pred_box, gt_box)
                    
                    if iou >= self.iou_threshold and not gt_detected[gt_idx] and not pred_matched[pred_idx]:
                        # 检测到真实病灶
                        gt_detected[gt_idx] = True
                        pred_matched[pred_idx] = True
                        total_tp += 1
                        break
            
            # 计算假阳性
            total_fp += sum(1 for matched in pred_matched if not matched)
        
        # 计算敏感度和每患者假阳性率
        sensitivity = total_tp / total_gt if total_gt > 0 else 0
        fp_per_patient = total_fp / n_patients if n_patients > 0 else 0
        
        return fp_per_patient, sensitivity
    
    def plot_froc_curves(self, results, output_dir, case_results=None):
        """Plot FROC curves"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Use a basic font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        
        plt.figure(figsize=(12, 8))

        # Plot FROC curves for different lesion types
        lesion_type_names = {
            'index': 'Index Lesions',
            'clinical': 'Clinically Significant Lesions',
            'all': 'All Lesions'
        }

        for lesion_type, name in lesion_type_names.items():
            fps = np.array(results[lesion_type]['fps_list'])
            sens = np.array(results[lesion_type]['sens_list'])

            # Sort data points by FP rate
            sort_idx = np.argsort(fps)
            fps_sorted = fps[sort_idx]
            sens_sorted = sens[sort_idx]

            plt.plot(fps_sorted, sens_sorted, label=name, linewidth=2)

            # Mark sensitivity at 1 FP/patient
            sens_at_1fp = self._interpolate_at_fp(fps, sens, 1.0)
            plt.plot(1.0, sens_at_1fp, 'o', markersize=8)
            plt.text(1.1, sens_at_1fp, f'{sens_at_1fp:.1%}', fontsize=10)

        # Add vertical line marking 1 FP/patient
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)

        # Set chart properties
        plt.xlabel('False Positives per Patient (FP/patient)', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.title('FROC Analysis Curves', fontsize=14)
        plt.xlim([0, 2])
        plt.ylim([0, 1.0])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Save the chart
        plt.savefig(f"{output_dir}/froc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Only plot Gleason score group curves if case_results is provided
        if case_results is not None:
            self._plot_gs_froc_curves(case_results, output_dir)
    
    def _plot_gs_froc_curves(self, case_results, output_dir):
        """Plot FROC curves for different Gleason score groups"""
        # Define Gleason score groups
        gs_groups = {
            'GGG1': [0],
            'GGG2': [1],
            'GGG3': [2],
            'GGG4': [3],
            'GGG5': [4]
        }
        
        plt.figure(figsize=(12, 8))
        
        # Threshold range
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        # Plot FROC curves for each Gleason score group
        for gs_name, gs_classes in gs_groups.items():
            fps_list = []
            sens_list = []
            
            for threshold in thresholds:
                fps, sens = self._evaluate_gs_at_threshold(case_results, gs_classes, threshold)
                fps_list.append(fps)
                sens_list.append(sens)
            
            # Sort data points by FP rate
            fps = np.array(fps_list)
            sens = np.array(sens_list)
            sort_idx = np.argsort(fps)
            fps_sorted = fps[sort_idx]
            sens_sorted = sens[sort_idx]
            
            plt.plot(fps_sorted, sens_sorted, label=gs_name, linewidth=2)
            
            # Mark sensitivity at 0.5 and 1 FP/patient
            for fp_mark in [0.5, 1.0]:
                sens_at_fp = self._interpolate_at_fp(fps, sens, fp_mark)
                plt.plot(fp_mark, sens_at_fp, 'o', markersize=6)
                plt.text(fp_mark+0.1, sens_at_fp, f'{sens_at_fp:.1%}', fontsize=8)
        
        # Add vertical lines marking 0.5 and 1 FP/patient
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Set chart properties
        plt.xlabel('False Positives per Patient (FP/patient)', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.title('FROC Analysis Curves for Different Gleason Grade Groups', fontsize=14)
        plt.xlim([0, 2])
        plt.ylim([0, 1.0])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save the chart
        plt.savefig(f"{output_dir}/gs_froc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, metrics, output_dir):
        """Generate performance report"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create table for lesion type metrics (as shown in the image)
        lesion_metrics_data = {
            'Model': ['Your Model'],  # Replace with your model name
            'Index Lesions FP@Sen80%': [metrics.get('index_FP@Sen80%', 'N/A')],
            'Index Lesions FP@Sen90%': [metrics.get('index_FP@Sen90%', 'N/A')],
            'Clinically Significant Lesions FP@Sen80%': [metrics.get('clinical_FP@Sen80%', 'N/A')],
            'Clinically Significant Lesions FP@Sen90%': [metrics.get('clinical_FP@Sen90%', 'N/A')],
            'All Lesions FP@Sen60%': [metrics.get('all_FP@Sen60%', 'N/A')],
            'All Lesions FP@Sen80%': [metrics.get('all_FP@Sen80%', 'N/A')]
        }
        
        lesion_metrics_df = pd.DataFrame(lesion_metrics_data)
        lesion_metrics_df.to_csv(f"{output_dir}/lesion_type_metrics.csv", index=False)
        
        # Create formatted table similar to the image
        formatted_data = []
        model_name = "Your Model"  # Replace with your model name
        
        formatted_data.append({
            'Model': model_name,
            'Index FP@Sen80%': f"{metrics.get('index_FP@Sen80%', 0):.3f}",
            'Index FP@Sen90%': f"{metrics.get('index_FP@Sen90%', 0):.3f}",
            'Clinical FP@Sen80%': f"{metrics.get('clinical_FP@Sen80%', 0):.3f}",
            'Clinical FP@Sen90%': f"{metrics.get('clinical_FP@Sen90%', 0):.3f}",
            'All FP@Sen60%': f"{metrics.get('all_FP@Sen60%', 0):.3f}",
            'All FP@Sen80%': f"{metrics.get('all_FP@Sen80%', 0):.3f}"
        })
        
        formatted_df = pd.DataFrame(formatted_data)
        formatted_df.to_csv(f"{output_dir}/formatted_lesion_metrics.csv", index=False)
        
        # Create HTML version with better formatting
        html_output = f"""
        <html>
        <head>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    text-align: center;
                    padding: 8px;
                    border: 1px solid black;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h2>Performance Metrics at Different Sensitivity Levels</h2>
            <table>
                <tr>
                    <th rowspan="2">Model</th>
                    <th colspan="2">Index lesions</th>
                    <th colspan="2">Clinically significant lesions</th>
                    <th colspan="2">All lesions</th>
                </tr>
                <tr>
                    <th>FP@Sen80%</th>
                    <th>FP@Sen90%</th>
                    <th>FP@Sen80%</th>
                    <th>FP@Sen90%</th>
                    <th>FP@Sen60%</th>
                    <th>FP@Sen80%</th>
                </tr>
                <tr>
                    <td>{model_name}</td>
                    <td>{metrics.get('index_FP@Sen80%', 0):.3f}</td>
                    <td>{metrics.get('index_FP@Sen90%', 0):.3f}</td>
                    <td>{metrics.get('clinical_FP@Sen80%', 0):.3f}</td>
                    <td>{metrics.get('clinical_FP@Sen90%', 0):.3f}</td>
                    <td>{metrics.get('all_FP@Sen60%', 0):.3f}</td>
                    <td>{metrics.get('all_FP@Sen80%', 0):.3f}</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        with open(f"{output_dir}/lesion_metrics_table.html", "w") as f:
            f.write(html_output)
        
        # Original GGG metrics table
        table2_data = {
            'Lesion Type': [],
            'Metric': [],
            'Value': []
        }

        # Add metrics for each Gleason score group
        gs_groups = ['GGG1', 'GGG2', 'GGG3', 'GGG4', 'GGG5']
        for gs in gs_groups:
            for metric_suffix in ['FP@Sen60%', 'FP@Sen80%', 'FP@Sen90%']:
                key = f"{gs}_{metric_suffix}"
                if key in metrics:
                    table2_data['Lesion Type'].append(gs)
                    table2_data['Metric'].append(metric_suffix)
                    table2_data['Value'].append(metrics[key])

        table2_df = pd.DataFrame(table2_data)
        table2_df.to_csv(f"{output_dir}/gleason_score_metrics.csv", index=False)

        # Create summary of key results
        with open(f"{output_dir}/summary_results.txt", 'w') as f:
            f.write("======== FROC Analysis Results ========\n\n")

            f.write("Key Performance Metrics at 1 FP/patient:\n")
            f.write(f"• Index Lesions: {metrics.get('index_Sen@FP1', 'N/A'):.1%} sensitivity\n")
            f.write(f"• Clinically Significant Lesions: {metrics.get('clinical_Sen@FP1', 'N/A'):.1%} sensitivity\n")
            f.write(f"• All Lesions: {metrics.get('all_Sen@FP1', 'N/A'):.1%} sensitivity\n")

        # 创建二分类 ROC AUC 结果的 DataFrame
        roc_auc_data = {
            'Classification': ['GS≥7 vs GS<7', 'GS≥4+3 vs GS≤3+4'],
            'ROC AUC': [
                f"{metrics.get('GS≥7_vs_GS<7', 'N/A'):.3f}",
                f"{metrics.get('GS≥4+3_vs_GS≤3+4', 'N/A'):.3f}"
            ]
        }
        
        roc_auc_df = pd.DataFrame(roc_auc_data)
        roc_auc_df.to_csv(f"{output_dir}/binary_roc_auc_metrics.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Perform FROC analysis for lesion detection.')
    parser.add_argument("--pkl_dir", type=str, required=True,
                        help="Path to the directory containing model prediction files")
    parser.add_argument("--gt_npz_dir", type=str, required=True,
                        help="Path to the directory containing ground truth files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output results")
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="Confidence threshold for predictions. Default: 0.2")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching. Default: 0.5")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level. Default: INFO")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    # Initialize analyzer
    analyzer = FROCAnalyzer(iou_threshold=args.iou_threshold)
    
    # Load data
    logger.info("Loading and preprocessing data...")
    case_results = analyzer.load_data(
        args.gt_npz_dir, 
        args.pkl_dir,
        confidence_threshold=args.confidence_threshold
    )
    
    if not case_results:
        logger.error("No valid data found. Exiting.")
        sys.exit(1)
    
    # Calculate FROC metrics
    logger.info("Computing FROC metrics...")
    results, metrics, case_results = analyzer.evaluate_froc(case_results)
    
    # Output key metrics
    logger.info("\n====== Key Performance Metrics ======")
    logger.info(f"Index Lesion Sensitivity (1 FP/patient): {metrics['index_Sen@FP1']:.1%}")
    logger.info(f"Clinically Significant Lesion Sensitivity (1 FP/patient): {metrics['clinical_Sen@FP1']:.1%}")
    logger.info(f"All Lesion Sensitivity (1 FP/patient): {metrics['all_Sen@FP1']:.1%}")
    
    # Generate FROC curves
    logger.info("Generating FROC curves...")
    analyzer.plot_froc_curves(results, args.output_dir, case_results)
    
    # Generate reports
    logger.info("Generating performance reports...")
    analyzer.generate_report(metrics, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()