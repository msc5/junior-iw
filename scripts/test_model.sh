
python -m src test $1 $2 \
	--checkpoint_path "results/train/${1}_${2}/${3}/checkpoints/final.ckpt" \
	--num_layers $3 \
	--task_id $3 \
	--seq_len 50
