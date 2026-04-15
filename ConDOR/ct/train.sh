# python main.py \
#   --warmup 1 \
#   --dir /home/nvidia-lab/ai4life/thaind2/brain/ConDOR-ICLR25/ConDOR/ct/preprocess_data/data4condor \
#   --ct_csv CT_train.csv \
#   --num_node 68 \
#   --classes 3 \
#   --data_min 1.138 --data_max 4.307 \
#   --age_min 55.1 --age_max 93.1 \
#   --batch 16 \
#   --warmup_num_steps 10000 \
#   --train_num_steps 10000 \
#   --test_num 141


python main.py \
  --warmup 0 \
  --dir /home/nvidia-lab/ai4life/thaind2/brain/ConDOR-ICLR25/ConDOR/ct/preprocess_data/data4condor \
  --ct_csv CT_train.csv \
  --num_node 68 \
  --classes 3 \
  --data_min 1.138 --data_max 4.307 \
  --age_min 55.1 --age_max 93.1 \
  --batch 16 \
  --train_num_steps 10000 \
  --test_num 141