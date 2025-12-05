#!/bin/bash
set -euo pipefail

# ===== 基础环境 =====
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# 默认训练配置
WORK_DIR="output"
default_task="exp"

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
SAVE_DIR="${WORK_DIR}/${MLP_TASK_ID:-default_task}"
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
  --batch_size 48
  --save_results_interval 500
  --save_ckpt_interval 1000
  --output_dir "$SAVE_DIR"
  --num_train_steps 1000000
  --dataloader_num_workers 16
  --lr 1e-4
  --codebook_size 8192
  --codebook_embed_dim 256
  --codebook_l2_norm
  --codebook_show_usage
  --commit_loss_weight 0.2
  --image_size 512 512
  --reconstruction_weight 1.0
  --codebook_weight 1.0
  --gradient_accumulation_steps 1
)

# ==== 启动训练脚本 ====
set -x
accelerate launch --config_file accelerate_config.yaml \
    --multi_gpu \
    trainer/train_dino_tokenizer_imagenet.py \
    "${DEFAULT_ARGS[@]}" \
    "${PASSTHRU_ARGS[@]}"    
