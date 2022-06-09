import tensorflow as tf
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Cifar10 Prediction")
parser.add_argument("--input",type=str,help='image PATH')
parser.add_argument("--model",type=str,default='CNN')

args = parser.parse_args()

if args.model == 'mlp':
    model = tf.keras.models.load_model('D:\Assignment43-python\Cifar10\MLP_Cifar10.h5')

else:
    model = tf.keras.models.load_model('MLP_CNN_Cifar10.h5')


img = cv2.imread(args.input)
img = cv2.resize(img, (32,32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img/255.0
img = img.reshape(32,32,3)

# result
result = np.argmax(model.predict(img))
names_img = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(names_img(result))



