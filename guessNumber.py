import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ctypes  # An included library with Python install.

text = "press q to quit\npress r to reset\npress g to guess\nTry to draw in the center"
ctypes.windll.user32.MessageBoxW(0, text, "Game Controls", 0)

def click_event(event, x, y, flags, param):
    """
    This function detects the mouse clicking.
    Left Mouse Clicked - makes the pixel white
    Right Mouse Clicked - makes the pixel black
    """

    origX = x // 5
    origY = y // 5
    if flags == cv2.EVENT_FLAG_LBUTTON:
        screen[origY][origX] = 1
        bigScreen[origY * 5:(origY + 1) * 5, origX * 5:(origX + 1) * 5] = np.ones((5, 5), dtype=np.float)
    elif flags == cv2.EVENT_FLAG_RBUTTON:
        screen[origY][origX] = 0
        bigScreen[origY * 5:(origY + 1) * 5, origX * 5:(origX + 1) * 5] = np.zeros((5, 5), dtype=np.float)

def view_classify(img, ps):
    """
    Function for viewing an image and it's predicted classes.
    """

    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap="gray")
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

#loading the model
model = torch.load("model.pt")
model.eval()

#creating 28 X 28 image and enlarging it for better view
screen = np.zeros((28, 28), dtype=np.float)
bigScreen = cv2.resize(screen, (140, 140))
cv2.rectangle(bigScreen, (30, 20), (110, 120), 0.5, 2)
cv2.imshow("board", bigScreen)

while True:

    cv2.setMouseCallback("board", click_event)
    cv2.imshow("board", bigScreen)

    key = cv2.waitKey(1)
    if key == ord('q'):                      # press q to quit the game
        break
    if key == ord('r'):                      # press r to reset the game
        screen = np.zeros((28, 28), dtype=np.float)
        bigScreen = cv2.resize(screen, (140, 140))
        cv2.rectangle(bigScreen, (30, 20), (110, 120), 0.5, 2)
    if key == ord('g'):                      # press g to guess the number drawn
        inputFeature = screen.reshape((1, 784))
        inputFeature = torch.from_numpy(inputFeature).float()
        with torch.no_grad():
            logps = model(inputFeature)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        view_classify(inputFeature.view(1, 28, 28), ps)

cv2.destroyAllWindows()

