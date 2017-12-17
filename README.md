# My-Tacotron

## Download Dataset

[LJ Speech Dataset](https://drive.google.com/a/columbia.edu/file/d/1kv4ij1sj2hP4AFc5vP88ZW6Quj7u4duq/view?usp=sharing)

## Data preprocessing
Preprocess the raw data for easy pipeline in training

1. Unzip the Dataset
2. Edit the data path in `preprocess.py`.(default `./`)
3. Run `python3 preprocess.py`. This will generate `./training/` with a text as index and np files to store the preprocessed data

## Train
Train a model

1. Edit the preprocessed data path in `train.py`.(default `./training`)
2. If you want load pre train model, edit the checkpoint path and change `is_restore` to be True
3. Edit the parameters in `hparams.py` if needed
4. Run `python3 train.py`

## Evaluation

Generate .wav file using the model and random texts from speeches
Generate a file to store the input sentences

1. Create `./test` folder
2. Edit the number of tests you want in `eval.py` (default 10 to save time)
3. Edit the checkpoint path and output path if needed. (default `./checkpoint/model.ckpt-894000` and `./test`)
4. Run `python3 eval.py`

## Rating

Speech recognize the generated wav and compare it with raw input, giving an accuracy

1. Run `python test.py`
