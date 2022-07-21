from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import qimage2ndarray
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('D:\Face Mask Detection\Face_Mask.h5')

class FaceMask(QMainWindow):

    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.setWindowTitle('Face Mask Detection')
        self.ui = loader.load('form.ui',None)
        self.width = 224
        self.height = 224
        self.capture = cv2.VideoCapture(0)

        while True:
            ret,frame = self.capture.read()
            
            if ret == False:
                break

            image = qimage2ndarray.array2qimage(frame.data)  #SOLUTION FOR MEMORY LEAK
            self.ui.label_cam.setPixmap(QPixmap.fromImage(image))
            cv2.waitKey(1)

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img = cv2.resize(image ,(self.width,self.height))
            img = img / 255
            img = img.reshape(1,self.width,self.height,3) 

            pred = model.predict(img)
            result = np.argmax(pred)
            org = (35,45)
            fontScale = 1
            color = (255,0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if result == 0:
                cv2.putText(frame,'With Mask',org,font,fontScale,color,4)
    
            elif result == 1:
                cv2.putText(frame,'Without Mask',org,font,fontScale,color,4)
                
            if cv2.waitKey(1) == ord("q"):
                break
    

            cv2.imshow('frame',frame)

            # cv2.waitKey(1)  #ejra 10ms


app = QApplication([])

window = FaceMask()

app.exec()
