# yam-peg-insert

基于 Google DeepMind [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（`i2rt_yam`）与 `mujoco.MjSpec` 构建的 **YAM 机械臂精密插销（Peg-in-Hole）** 开源具身智能项目。

专为 **Ubuntu 22.04（ROS 2 Humble）+ 无独立显卡（纯 CPU 调试）+ 云端 GPU 训练** 设计，完整覆盖四周实战路线图：

1. **本地隔离环境**：通过 `uv` 创建 Python 3.12 虚拟环境，完全不干扰 ROS 2 Humble 的系统 Python 3.10。
2. **自定义仿真环境**：基于 `MjSpec` 动态拼接 `i2rt_yam`、工作台、凸多边形孔座与插销，避免非凸网格碰撞穿透。
3. **雅可比 IK 示教专家**：内置阻尼最小二乘逆运动学控制器（`oracle.py`），在无硬件、无训练状态下即可演示插销并采集专家轨迹。
4. **云端 RL 与远程推理**：支持 Google Colab GPU 并行训练 PPO，并通过轻量级 HTTP 推理桥接实现“云端策略出动作 → 本地 MuJoCo 执行”。

## 1. Ubuntu 22.04 一键初始化

```bash
bash scripts/setup_ubuntu22.sh
source .venv/bin/activate
```

如在无独立显卡设备上运行交互式可视化遇到 OpenGL 上下文报错，可先设置软渲染环境变量：

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

## 2. 本地 CPU 常用命令

```bash
# 1. 无头物理冒烟测试
python scripts/smoke_test.py

# 2. 启动 MuJoCo 交互式窗口查看场景
python scripts/view_scene.py

# 3. 运行内置雅可比 IK 专家演示（可选 --record 保存演示轨迹到 demos/）
python scripts/run_oracle.py --episodes 3 --record

# 4. 运行单元测试
pytest
```

## 3. 云端 GPU 训练与本地回放

1. 将仓库推送到 GitHub 后，在 Google Colab（开启 T4 GPU）运行 `notebooks/train_colab_ppo.ipynb`。
2. 将训练好的模型权重（如 `ppo_yam_peg_insert.zip`）下载到本地 `checkpoints/` 目录。
3. 在本地 Ubuntu 22.04 CPU 上评估或可视化回放：

```bash
python scripts/eval_policy.py --checkpoint checkpoints/ppo_yam_peg_insert.zip --episodes 10
```

## 4. 远程策略推理桥接（云端模型 ↔ 本地仿真）

```bash
# 云端（或服务端）启动策略服务
python -m yam_peg_insert.remote_policy --mode server --port 8765

# 本地 Ubuntu 22.04 启动仿真客户端执行动作
python -m yam_peg_insert.remote_policy --mode client --url http://127.0.0.1:8765 --episodes 2
```
