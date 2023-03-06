#### This project trains a couplet prediction model (the prediction model only supports 14-character couplets) using teacher-forcing.

When giving the model an input of the first half of the couplet (7 Chinese characters), it will predict the second half.
Some ideas & block implementations of the GPT, (especially the positional encoding part), are based on this great 
Transformer notebook from Harvard: https://nlp.seas.harvard.edu/2018/04/03/attention.html

Get the dataset: https://github.com/wb14123/couplet-dataset. Put all the txt files under a single folder "Dataset", under 
the project directory, to run the script

utils.py: defines the GPT model and some helper functions

train.py: configuration setting and training. After running this script, a .pickle file containing the word dict and a .pt model
will be saved to the configured paths. Each time you run the script, the order of the word dict will be different. 

predict.py: after running train.py and getting the word dict & model, set the "sen_str" variable in predict.py (the first half 
of a couplet), and run the script to get the second half as output.

The trained model can be accessed at: https://drive.google.com/file/d/132E2gSzpjOpdWz0Qo-1MgzeUAnq8a59I/view?usp=sharing

The model's corresponding word dict can be accessed at: https://drive.google.com/file/d/1h3wUJ2Dw0tfleUMve4jVlnBl2d-ZywII/view?usp=sharing
