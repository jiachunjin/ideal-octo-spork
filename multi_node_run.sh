WORLD_SIZE=2
SHARED_DIR="/data/phd/jinjiachun/shared_dir"

ACCELERATE_CONFIG_FILE="config/accelerate_config/multi_node_example_zero2.yaml"
MAIN_PYTHON_SCRIPT="runner/ar_backbone/train_diffhead.py"
MAIN_PYTHON_SCRIPT_ARGS="config/ar_backbone/diffhead_and_ar.yaml"

# 解析命令行参数
for ARG in "$@"; do
  case $ARG in
    --WORLD_SIZE=*)
      WORLD_SIZE="${ARG#*=}"
      shift
      ;;
    --config_path=*)
      CONFIG_PATH="${ARG#*=}"
      shift
      ;;
    --SHARED_DIR=*)
      SHARED_DIR="${ARG#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $ARG"
      exit 1
      ;;
  esac
done

# 如果环境变量 INDEX 没有设置，则设为 0，并确保 WORLD_SIZE 为 1
if [ -z "$INDEX" ]; then
  INDEX=0
  WORLD_SIZE=1
fi

# export NCCL_DEBUG=INFO
HOSTNAME=$MY_NAME
IP=$(hostname -I | awk '{print $1}')
echo "当前机器IP地址是: $IP"

NODE_RANK=$INDEX
NUM_GPUS_PER_NODE=$MY_GPU
TRIAL_ID=$KML_ID

# master 信息文件（所有机器共享）
MASTER_FILE=$SHARED_DIR/master_${TRIAL_ID}.txt

# master 生成 master 文件（仅 INDEX=0 执行）
if [[ "$NODE_RANK" == "0" ]]; then
    while true; do
        PORT=29509
        if ! lsof -i:$PORT &>/dev/null; then
            echo "${IP} ${PORT}" > $MASTER_FILE
            echo "[INFO] Master node info written to $MASTER_FILE: $IP:$PORT"
            break
        fi
    done
else
    while [ ! -f $MASTER_FILE ]; do
        echo "[INFO] Waiting for master file: $MASTER_FILE"
        sleep 1
    done
fi

# 所有节点读取 master 地址和端口
read MASTER_ADDR MASTER_PORT < $MASTER_FILE

# echo "[INFO] Starting accelerate with world size $WORLD_SIZE, gpus per node $NUM_GPUS_PER_NODE, master_addr=$MASTER_ADDR, master_port=$MASTER_PORT, node_rank=$NODE_RANK"
# echo "[INFO] Using config: $CONFIG_PATH"
# echo "[INFO] Shared directory: $SHARED_DIR"

NUM_PROCESSES=$((NUM_GPUS_PER_NODE * WORLD_SIZE))
echo "[INFO] Total processes to launch: $NUM_PROCESSES (calculated from ${NUM_GPUS_PER_NODE} GPUs/node * ${WORLD_SIZE} machines)"

echo "[INFO] Starting accelerate launch with the following parameters:"
echo "  --config_file ${ACCELERATE_CONFIG_FILE}"
echo "  --main_process_ip ${MASTER_ADDR}"
echo "  --main_process_port ${MASTER_PORT}"
echo "  --machine_rank ${NODE_RANK}"
echo "  --num_machines ${WORLD_SIZE}"
echo "  --num_processes ${NUM_PROCESSES}"
echo "  Python Script: ${MAIN_PYTHON_SCRIPT}"
echo "  Script Arguments: ${MAIN_PYTHON_SCRIPT_ARGS}"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG_FILE" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --machine_rank "$NODE_RANK" \
  --num_machines "$WORLD_SIZE" \
  --num_processes "$NUM_PROCESSES" \
  "$MAIN_PYTHON_SCRIPT" \
  $MAIN_PYTHON_SCRIPT_ARGS

echo "[INFO] accelerate launch command finished."