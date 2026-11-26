# yam-peg-insert

基于 Google DeepMind [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（`i2rt_yam`）与 `mujoco.MjSpec` 构建的 **YAM 机械臂精密插销（Peg-in-Hole）** 轻量级仿真与强化学习仓库。

专为 **Ubuntu 22.04（ROS 2 Humble）+ 无独立显卡（纯 CPU 调试）+ 云端 GPU 训练** 设计。

## 1. 本地环境配置（不破坏 ROS 2 Humble 的 Python 3.10）

请使用 `uv` 在独立虚拟环境中安装，**不要**使用系统 `pip`：

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库并创建 Python 3.12 隔离环境
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 2. 本地 CPU 快速验证

```bash
# 无头物理冒烟测试（验证 MjSpec 模型拼接与 Gymnasium 环境）
python scripts/smoke_test.py

# 启动 MuJoCo 交互式窗口查看 YAM + 插销 + 孔座场景
# 如遇无独显 OpenGL 报错，可先执行: export LIBGL_ALWAYS_SOFTWARE=1
python scripts/view_scene.py

# 运行单元测试
pytest
```

## 3. 云端 GPU 训练（Google Colab）

将本仓库推送到 GitHub 后，在 Google Colab 中打开 `notebooks/train_colab_ppo.ipynb`（开启 T4 GPU），即可拉取仓库并启动 PPO 训练。
