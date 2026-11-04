# port_align_field

在 Chongyu 的 ManiSkill 分支（`wminsertion`，提交 7a0abea，代码未改）上，验证一个 SID 式对齐场能否把分布外的插头–插座状态拉进分布内，并在没见过的插座形状、握持俯仰角和握持位置下保持有效。方法、数学、文献依据和评估方案见 [docs/plan.md](docs/plan.md)。

## 目录
- `paf/env_adapter.py` 唯一调用环境的地方（重置、观测、移动手、直推、专家插一次）
- `paf/geometry.py` 插座坐标系和相对状态 q
- `paf/collect.py` 每个握持：插一次 → 退到插入前位姿 → 扫偏差 → 直推试验
- `paf/perception.py` 点云（RANSAC 板面、插座坐标系）和 RGB 裁剪
- `paf/model.py` 点云给 6D、RGB 给亚毫米残差的网络
- `paf/field.py` 场：能量、mean-shift 场、学习的场 f_θ
- `paf/train.py` 训练一组（留一种形状）
- `paf/policy.py` 闭环中方法一侧（不接触任何真值）
- `paf/evaluate.py` 闭环测试与打分
- `scripts/` 启动、监控、汇总；`tests/` 坐标代数和"真值不泄漏"检查

## 运行（服务器）
```bash
source scripts/env.sh          # venv + GPU Vulkan（AutoDL 文档的 EGL ICD 方案）
bash scripts/collect.sh 15     # 训练数据，可断点续跑
bash scripts/train.sh 60       # 四组同时训练
bash scripts/eval.sh 15        # 闭环评估 + 汇总表 results/summary.json
bash scripts/monitor.sh 20     # CPU/内存/GPU 逐秒采样
python -m pytest -q tests
```

## 环境
- `/root/autodl-tmp/paf_venv`（系统 torch 2.8 + numpy 1.26.4、sapien 3.0.0b1、mplib 0.2.1、`deps/ManiSkill` 可编辑安装）
- `/etc/vulkan/icd.d/my_nvidia_icd.json` 指向 `libEGL_nvidia.so.0`（AutoDL 帮助文档"最佳实践 → Vulkan"）
- 数据 `data/`、模型 `runs/`、日志 `logs/` 不进 git
