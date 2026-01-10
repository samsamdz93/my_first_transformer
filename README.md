# Transformer for Text Translation

French to english translator based on a transformer architecture. The transformer was implemented from scratch.

My best model has a 0.5 training loss and 2.2 validation loss. The accuracies are 89% and 65%.
I have some issues with the `<EOS>` token : my model doesn't finish its generation. Other updates will debug this.

## Dataset used

The dataset used is in the folder `data`. It contains 130k french sentences and their translation in english.

## Start interaction with a model

`python3 text_generation.py` will open a prompt to start interaction with the model. If an error occurs, it's probably because of the file path of the model.

## Training a model

Here is an example of command to train a new model
`python3 translator.py --dataset_path ./data --model_path ./models/model.pth --lr 0.0002 --batch_size 16 --momentum 0.7 --save_dir results --nepochs 10 --label_smoothing 0.1 --dropout 0.1` 

## Files description

 * `text_generation.py` : Runs the code that generates the prompt and start interaction with the model
 * `dataset_manager.py` : Preprocess the dataset.
 * `translator.py` : Main code for training a new model. All hyperparameters of the model must be written on the command line ; the file will read them using argparse.
 * `train_model.py` : Contains the training loop.
 * `neural_networks` (folder) : Implementation of all neural networks (transformer, decoder, encoder, attention).
 * `results/` : Training results of my best model.

