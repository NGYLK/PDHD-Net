import argparse
import functools
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nndet.io import load_pickle, save_json
from nndet.utils.info import maybe_verbose_iterable

# 为每个 print 添加 flush=True 以确保日志实时输出
print = functools.partial(print, flush=True)


def atomic_image_write(
    image: sitk.Image,
    path: Path,
    backup_existing_file: bool = False,
    compress: bool = True,
    mkdir: bool = False
):
    """
    以原子方式保存 SimpleITK 图像。
    参数说明保持不变
    """
    path = Path(path)

    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图像到临时文件
    path_tmp = path.with_name(f"tmp_{path.name}")
    sitk.WriteImage(image, path_tmp.as_posix(), useCompression=compress)

    # 备份现有文件
    if backup_existing_file and path.exists():
        dst_path_bak = path.with_name(f"backup_{path.name}")
        if dst_path_bak.exists():
            raise FileExistsError(f"已有备份文件存在于 {dst_path_bak}.")
        path.rename(dst_path_bak)

    # 重命名临时文件为目标文件
    path_tmp.rename(path)


def boxes2det(in_dir_pred, out_dir_det, threshold=0.0, min_num_voxels=10, return_detection_maps=True, verbose=1):
    """
    将 nnDetection 的边界框转换为多个标签的非重叠检测图。
    参数说明保持不变
    """
    # 设置输入输出路径
    in_dir_pred = Path(in_dir_pred)
    out_dir_det = Path(out_dir_det)
    out_dir_det.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
        生成检测图
        输入目录: {in_dir_pred}
        输出目录: {out_dir_det}
        最低置信度: {threshold}
        最小体素数: {min_num_voxels}
        """)

    # 获取所有案例 ID（假设文件名格式为 {case_id}_boxes.pkl）
    case_ids = [p.stem.rsplit('_', 1)[0] for p in in_dir_pred.glob("*_boxes.pkl")]
    if not case_ids:
        raise ValueError("输入目录中未找到任何边界框文件 (*.pkl)")
    else:
        print(f"找到 {len(case_ids)} 个案例") if verbose else None

    y_det = {}
    prediction_meta = {}
    for cid in maybe_verbose_iterable(case_ids):
        case_prediction_meta = {}
        res = load_pickle(in_dir_pred / f"{cid}_boxes.pkl")

        original_size = res["original_size_of_raw_data"]
        if len(original_size) == 2:
            original_size = (*original_size, 1)  # 确保为 3D

        # 初始化每个标签的 mask
        labels_present = np.unique(res["pred_labels"])
        labels_present = labels_present[labels_present >= 0]  # 排除负标签（如果有）
        label_masks = {label: np.zeros(original_size, dtype=float) for label in labels_present}

        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        # 过滤低置信度的边界框
        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        # 按置信度从高到低排序
        idx = np.argsort(scores)[::-1]
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
            # 跳过不存在的标签
            if plabel not in label_masks:
                continue

            # 保存边界框元数据
            case_prediction_meta[instance_id] = {
                "score": float(pscore),
                "label": int(plabel),
                "box": list(map(int, pbox))
            }

            # 构建 3D 体素的切片
            mask_slicing = [
                slice(int(pbox[0]) + 1, int(pbox[2])),
                slice(int(pbox[1]) + 1, int(pbox[3])),
            ]
            neighbourhood_slicing = [
                slice(max(0, int(pbox[0]) - 1), int(pbox[2]) + 2),
                slice(max(0, int(pbox[1]) - 1), int(pbox[3]) + 2),
            ]
            if len(original_size) == 3:
                mask_slicing.append(slice(int(pbox[4]) + 1, int(pbox[5])))
                neighbourhood_slicing.append(slice(max(0, int(pbox[4]) - 1), int(pbox[5]) + 2))

            mask_slicing = tuple(mask_slicing)
            neighbourhood_slicing = tuple(neighbourhood_slicing)

            # 检查病变候选区域的体素数
            num_voxels = label_masks[plabel][mask_slicing].size
            if num_voxels <= min_num_voxels:
                continue

            # 检查是否与已有的高置信度区域重叠
            if np.max(label_masks[plabel][neighbourhood_slicing]) == 0:
                if verbose >= 2:
                    print(f"设置标签 {plabel} 的框 {mask_slicing} 置信度为 {pscore}")
                label_masks[plabel][mask_slicing] = pscore
            else:
                previous_p = np.max(label_masks[plabel][mask_slicing])
                if verbose >= 2:
                    print(f"标签 {plabel} 的框 {mask_slicing} 与置信度为 {previous_p:.4f} 的框重叠，已跳过")
                continue  # 直接跳过重叠的框

        # 转换并保存每个标签的检测图
        for label, mask in label_masks.items():
            if not np.any(mask):
                # 如果该标签的 mask 全为零，则跳过不保存
                if verbose >= 2:
                    print(f"标签 {label} 没有检测到病灶，不生成检测图。")
                continue

            if return_detection_maps:
                if cid not in y_det:
                    y_det[cid] = {}
                y_det[cid][label] = sitk.GetImageFromArray(mask)
                y_det[cid][label].SetOrigin(res["itk_origin"])
                y_det[cid][label].SetDirection(res["itk_direction"])
                y_det[cid][label].SetSpacing(res["itk_spacing"])

            # 转换为 SimpleITK 图像
            mask_itk = sitk.GetImageFromArray(mask)
            mask_itk.SetOrigin(res["itk_origin"])
            mask_itk.SetDirection(res["itk_direction"])
            mask_itk.SetSpacing(res["itk_spacing"])

            # 保存检测图，文件名中包含标签编号
            output_filename = out_dir_det / f"{cid}_detection_map_label{label}.nii.gz"
            atomic_image_write(mask_itk, output_filename, backup_existing_file=False, compress=True, mkdir=False)

        # 保存预测元数据
        save_json(case_prediction_meta, out_dir_det / f"{cid}_boxes.json")
        prediction_meta[cid] = case_prediction_meta

    if verbose:
        print("全部检测图生成完毕。")

    return prediction_meta, y_det


def main():
    """
    主函数，解析命令行参数并调用 boxes2det 函数。
    """
    parser = argparse.ArgumentParser(description='从 nnDetection 的边界框生成多标签的检测图')
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="包含模型预测结果（nnDetection 边界框）的文件夹路径")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="用于存储检测图的文件夹路径。默认：输入目录 + _detection_maps")
    parser.add_argument("-t", "--threshold", type=float, default=0.4,
                        help="最低置信度阈值。默认：0")
    parser.add_argument("-m", "--min_num_voxels", type=int, default=10,
                        help="病变候选区域的最小体素数。默认：10")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="冗余级别。1 为基本信息，2 为详细调试信息。默认：1")
    args = parser.parse_args()

    # 转换路径为 Path 对象
    input_dir = Path(args.input)
    if args.output is None:
        output_dir = input_dir.parent / (input_dir.name + "_detection_maps")
    else:
        output_dir = Path(args.output)

    # 执行转换
    boxes2det(
        in_dir_pred=input_dir,
        out_dir_det=output_dir,
        threshold=args.threshold,
        min_num_voxels=args.min_num_voxels,
        return_detection_maps=False,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()