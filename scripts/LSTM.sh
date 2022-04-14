
python -m src train LSTM GeneratedSins --num_layers 1 --max_epochs 2
python -m src train LSTM GeneratedSins --num_layers 2 --max_epochs 2

python -m src train LSTM GeneratedNoise --num_layers 1 --max_epochs 2
python -m src train LSTM GeneratedNoise --num_layers 2 --max_epochs 2
