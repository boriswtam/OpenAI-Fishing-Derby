Open ai gym final project 

Team name: unexpeced indent error 
Team member: Jacky Huang, Boris Tam, Quinn Coleman 

before runnning make sure you have Keras, Keras atari, tensorflow, and gym installed 

>pip3 install keras
>pip3 install keras[atri
>pip3 install tensorflow
>pip3 install gym

================================================================================

The enviornment we are using is https://gym.openai.com/envs/FishingDerby-v0/

================================================================================

For our project we ended up implementing 2 ways to train the model, one is to train the model with random actions, 
the other is the train the model with learned model from pervious iterations, 
with the first one generated from observing random play.                         

================================================================================
To train with random only

1. edit the paramenters as you would like such as gamma, epsilon, observetime and mb_size. 
2. Open up terminal
3. python3 atari.py

there is no graphic window displayed during training, however there will be progress print outs in the terminal
then once that is done training it should save a pair of files, named model.json and model.h5
------------------------------------------------------------------------
To train with reuse model
1. edit the paramenters as you would like such as gamma, epsilon, observetime and mb_size. 
2. Open up terminal
3. python3 atarifed.py

there is no graphic window displayed during training, however there will be progress print outs in the terminal
then once that is done training it should save a pair of files, named model.json and model.h5 

================================================================================

To test the mode 

1. Open up terminal 
2. python3 atariPlay.py 

this will load model.json and model.h5 into the enviornment and play the game with a trained model produced by 
one of the training methods above
