import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D
from tensorflow.keras import Model

class FaceNet(Model):
  def __init__(self):
    super().__init__()
    
    self.dense_1 = Dense(128,activation='relu')
    self.dense_2 = Dense(14,activation='softmax')
    self.dense_3 = Dense(32,activation='relu')
    self.conv2d_1 = Conv2D(64,(3,3),activation='relu',input_shape=(224, 224, 3))
    self.flatten = Flatten()
    self.conv2d_2 = Conv2D(64,(5,5),activation='relu')
    self.max_pool = MaxPool2D()

  def call(self,x):
    k = self.conv2d_1(x)
    y = self.max_pool(k)
    z = self.conv2d_2(y)
    u = self.flatten(y)
    c = self.dense_1(u)
    a = self.dense_3(c)
    out = self.dense_2(a)

    return out


  