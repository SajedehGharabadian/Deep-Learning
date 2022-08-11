import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from model import FaceNet

parser = argparse.ArgumentParser(description="Sodoku Detector")
parser.add_argument("--weights",type=str,help="path of your model")
parser.add_argument("--input",type=str,help="path of your input image")


args = parser.parse_args()

width = height = 224

model = FaceNet()
model.load_weights(args.weights)

img = cv2.imread(args.input)
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(image ,(width,height))
img = img / 255
img = img.reshape(1,width,height,3) 

result = np.argmax(model.predict(img))

face_persons= ["Ali Khamenei","Angelina Jolie","Barak Obama","Behnam Bani","Donald Trump","Emma Watson","Han Hye Jin","Kim Jong Un",
"Leyla Hatami","Lionel Messi","Michelle Obama","Morgan Freeman","Queen Elizabeth","Scarlett Johanson"]

print('This is : ',face_persons[result])    