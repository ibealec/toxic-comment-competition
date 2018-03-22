# Jigsaw/Kaggle Toxic Comment Competition (Ensemble Model)

This is my attempt at the toxic competition on Kaggle by Jigsaw.

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

I've organized all of the models that use word embeddings into corresponding iPython notebooks.

The notebooks are separated by their type of pretrained embedding models. Preprocessing can be found in *helpers.py* and models can be found in *architecture.py*.

I've taken my best results from every type of model and collected them into an ensemble, then applied the *Toxic Avenger* technique by the1owl. I do not recommend putting too much thought into the weights I've chosen, since they're based off of the success of my model with the Public Leaderboard, which could have a different distribution than the Private Leaderboard or any other dataset.