import tensorflow as tf
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Fashion Mnist Prediction")
parser.add_argument("--input",type=str,help='image PATH')
parser.add_argument("--model",type=str,default='CNN')

args = parser.parse_args()

if args.model == 'mlp':
    model = tf.keras.models.load_model('MLP_fashionMnist.h5')

else:
    model = tf.keras.models.load_model('CNN_MLP_fashionMnist.h5')


img = cv2.imread(args.input)
img = cv2.resize(img,(28,28))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = img / 255.0
img = img.reshape(28,28,1)

# result
result = np.argmax(model.predict(img))
print(result)



