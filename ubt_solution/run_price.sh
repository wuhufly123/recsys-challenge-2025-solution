LOG_NAME="train_features_epoch3_price"
OUTPUT_DIR="./log"  # 默认输出目录
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/$LOG_NAME$(date +%Y%m%d_%H%M%S).log"

# 可选种子设置
SEED=${SEED:-4}
echo "Using seed: $SEED"

python -m ubt_solution.create_embeddings \
    --data-dir /data_sdc/mhwang/recsys_code/data \
    --embeddings-dir /data_sdc/mhwang/recsys_code/submit_file/train_features_epoch3_price \
    --accelerator cuda \
    --devices 0 \
    --num-workers 4 \
    --batch-size 128 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --task-weights "churn:0.0,category_propensity:0.0,product_propensity:0.0,price:1.0" \
    --seed "$SEED" \
    > >(tee -a "$LOG_FILE") 2>&1
# 检查命令执行是否成功
if [ $? -eq 0 ]; then
    echo "训练完成，日志已保存至: $LOG_FILE"
else
    echo "训练失败，请检查日志: $LOG_FILE"
fi