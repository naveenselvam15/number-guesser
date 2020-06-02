# number-guesser
Guessing handwritten number using deep learning

Mnist dataset is present in dataset folder but we are going to play with binary images,
so cleaningData.py converts grayscale images into binary images and store it in csv 
files. I haven't uploaded the csv files because it is too large. After running 
cleaningData.py, csv files will be created in newDataset folder.Run trainingModel.py
which create, train, and test the model for classification using newModel folder 
and store the model as model.pt.

Reference links :

https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

https://www.youtube.com/watch?v=PXOzkkB5eH0
