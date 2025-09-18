# PDHD-Net 数据不平衡处理机制分析

## 🔍 核心发现
1. **PDHD-Net = nndetection + 改进的编解码器(C+SW+BIFPN)**
2. **nndetection内置完整的数据不平衡处理机制**
3. **只需修改1行代码即可启用FROC评估**

## 📊 数据不平衡处理策略

### 1. 数据采样层面
- `oversample_foreground_percent: 0.5` - 50%前景+50%背景
- `DataLoader3DBalanced` - 类别平衡采样
- 前景patch从预计算边界框采样，背景patch随机采样

### 2. 困难负样本挖掘
- `HardNegativeSamplerBatched` - 批次级采样
- `pool_size: 20` - 从20倍候选池选择困难负样本
- `positive_fraction: 0.33` - 33%正样本+67%负样本

### 3. IoU匹配策略
- 双阈值匹配：`low_threshold` + `high_threshold`
- 忽略区间避免边界模糊样本干扰
- `allow_low_quality_matches`确保每个GT有匹配

### 4. 损失函数
- 支持Focal Loss、交叉熵+权重、BCE+标签平滑
- 先验概率初始化：`prior_prob: 0.01`
- 可配置类别权重

## 🎯 FROC实现方案

### 启用方法
```python
# nndet/ptmodule/retinaunet/base.py 第110行
self.box_evaluator = BoxEvaluator.create(
    classes=_classes,
    fast=False,  # 🔥 改这里
    save_dir="./froc_output",
)
```

### FP=1敏感度获取
```python
fp1_idx = list(metric_curves["FROC_fpi_thresholds"]).index(1.0)  # 索引3
sensitivity_at_fp1 = metric_curves["FROC_curve_IoU_0.50"][fp1_idx]
```

### 预期输出
- 9个IoU阈值(0.1-0.9)的FROC指标
- 每类别独立FROC分析
- 自动生成可视化图表
- 关键指标：在平均1FP/患者下的敏感度

## 💡 实现评估
- **算法合理性**: ✅ 无投机取巧，理论基础扎实
- **工程质量**: ✅ 多层次防护，参数可调
- **实现难度**: ⭐ 极简单，改1-2行代码
- **医学适用性**: ✅ 专为医学影像设计

## 📋 实施清单
- [ ] 修改base.py启用FROC
- [ ] 可选：添加训练时FP显示
- [ ] 验证输出指标
- [ ] 分析FROC图表