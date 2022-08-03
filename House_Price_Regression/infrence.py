from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('D:\House_Price_Regression\house.h5')

inputImages = []
outputImage = np.zeros((64, 64, 3), dtype="uint8")

for i in range(4):
    image = cv2.imread('D:\House_Price_Regression\Infrence/'+str(i)+'.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    inputImages.append(image)

outputImage[0:32, 0:32] = inputImages[0]
outputImage[0:32, 32:64] = inputImages[1]
outputImage[32:64, 0:32] = inputImages[2]
outputImage[32:64, 32:64] = inputImages[3]

outputImage = outputImage / 255.0
# cv2.imshow(outputImage)
# cv2.imwrite('out.jpg',outputImage)
outputImage = outputImage.reshape(1, 64, 64, 3)
pred = model.predict([outputImage])
print('Estimated Price: ', pred[0])


