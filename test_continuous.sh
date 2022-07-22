python train.py \
  --device cuda:0 \
  --env LunarLanderContinuous-v2 \
  --log log_continuous \
  # --load "/home/xander/MPO/mpo/log_continuous/model/model_latest.pt" \
  --iteration_num 10 \
  --render