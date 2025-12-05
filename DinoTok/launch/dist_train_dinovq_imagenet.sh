#!/bin/bash
set -euo pipefail

# ===== 基础环境 =====
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# 默认训练配置
WORK_DIR="output"
default_task="dinovq_exp"

# ===== 输出目录：仅解析 --output_dir 用于 SAVE_DIR 拼接，其他参数全部透传 =====
# 从命令行获取参数，如果没有就使用默认
PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
        WORK_DIR="$2"
        shift 2
        ;;
        *)
        PASSTHRU_ARGS+=("$1")
        shift
        ;;
    esac
done

# 拼接保存路径（添加火山云任务 ID）
SAVE_DIR="${WORK_DIR}/${MLP_TASK_ID:-$default_task}"
echo "Saving to: $SAVE_DIR"

mkdir -p "$SAVE_DIR"

# ==== 读取火山云环境变量（自动填充） ====
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${MLP_WORKER_0_PORT:-22222}
NUM_GPUS=$(nvidia-smi -L | wc -l)
TOTAL_PROCESSES=$((NNODES * NUM_GPUS))

echo "Launching Accelerate training:"
echo " - Node rank: $NODE_RANK"
echo " - Master addr: $MASTER_ADDR"
echo " - Master port: $MASTER_PORT"
echo " - Num nodes: $NNODES"
echo " - GPUs per node: $NUM_GPUS"

# ==== 动态写入 accelerate 配置 ====
cat <<EOF > accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: true
deepspeed_config: null
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: ${NODE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
main_training_function: main
mixed_precision: 'no'
num_machines: ${NNODES}
num_processes: ${TOTAL_PROCESSES}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# ===== 训练的默认参数（可在此处统一改默认值）=====
DEFAULT_ARGS=(
  --batch_size 32
  --save_results_interval 200
  --save_ckpt_interval 1000
  --output_dir "$SAVE_DIR"
  --num_train_steps 1000000
  --dataloader_num_workers 16
  --lr 1e-4
  --codebook_size 8192
  --codebook_dim 256
  --img_size 512 512
  --gradient_accumulation_steps 1
  --use_perceptual
#   --use_gan
#   --disc_start 2000
#   --disc_weight 0.5
  --perceptual_weight 1.0
  --vq_loss_weight 1.0
  --recon_loss_weight 1.0
  --dino_loss_weight 1.0
)

# ==== 启动训练脚本 ====
set -x
accelerate launch --config_file accelerate_config.yaml \
    --multi_gpu \
    trainer/train_dinovq_imagenet.py \
    "${DEFAULT_ARGS[@]}" \
    "${PASSTHRU_ARGS[@]}"    

# ==============================
# 调用方式示例：
#
# 1. 默认启动 (带GAN和Perceptual Loss)
# bash launch/dist_train_dinovq_imagenet.sh
#
# 2. 修改GAN启动步数
# bash launch/dist_train_dinovq_imagenet.sh --disc_start 5000
#
# 3. 关闭GAN (需要修改脚本里的DEFAULT_ARGS或者覆盖参数，注意argparse的store_true行为)
#    注意：如果是store_true参数，脚本里默认写了就是开启。如果想关闭，需要在脚本里去掉 --use_gan
#
# 4. 自定义输出目录
# bash launch/dist_train_dinovq_imagenet.sh --output_dir output/my_experiment
# ==============================
