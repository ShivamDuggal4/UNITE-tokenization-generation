## Launches the job on 8 GPUs
TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=8 \
  --master_port=12342 main_train.py \
  --config configs/imagenet.yaml \
  --data-path $DATA_PATH \
  --results-dir outputs \
  --experiment-name lets-unite \
  --image-size 256