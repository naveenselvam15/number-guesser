# number-guesser
Guessing handwritten number using deep learning

Mnist dataset is present in dataset folder but we are going to play with binary images,
so cleaningData.py converts grayscale images into binary images and store it in csv 
files. I haven't uploaded the csv files because it is too large. After running 
cleaningData.py, csv files will be created in newDataset folder.Run trainingModel.py
which create, train, and test the model for classification using newModel folder 
and store the model as model.pt.

Run the guessNumber.py to begin the fun. It uses opencv functions to detect mouse clicks 
and draw on the screen. Then the image is send to the model and probabilities for each digits
are calculated and plotted.

GAME CONTROLS

'q' - quit the game
'r' - reset the game
'g' - guess the image 

Reference links :

https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

https://www.youtube.com/watch?v=PXOzkkB5eH0

https://www.youtube.com/watch?v=rrh-4NtuK-w&t=300s
