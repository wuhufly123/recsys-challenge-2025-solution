LOG_NAME="all_features"
OUTPUT_DIR="./baseline/baseline_logs"  # 默认输出目录
# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 设置日志文件路径
LOG_FILE="$OUTPUT_DIR/$LOG_NAME$(date +%Y%m%d_%H%M%S).log"

python -m baseline.aggregated_features_baseline.create_embeddings \
    --data-dir /data_sdc/mhwang/RecSys_sub/data_submit \
    --embeddings-dir /data_sdc/mhwang/RecSys_sub/Z_new_data/all_features \
    --num-days 1 7 14 30 90 180  \
    --top-n 10 \
    > >(tee -a "$LOG_FILE") 2>&1