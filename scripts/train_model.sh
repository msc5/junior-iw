
python -m src train $1 $2 \
	--num_layers $3 \
	--task_id $3 \
	--max_epochs 10 \
	--seq_len 50 \
	--batch_size 64 \
	--n_val_batches 0
