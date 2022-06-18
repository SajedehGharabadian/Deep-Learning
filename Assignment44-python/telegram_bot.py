import tensorflow as tf
import telebot
import cv2
import numpy as np

mybot = telebot.TeleBot("token")

@mybot.message_handler(commands=['start'])
def send_welcome(message):
    msg = mybot.send_message(message.chat.id,"Hi "+str(message.chat.first_name)+" welcome to mybot"+" \n"+
                            "/Recognition- Recognize Vehicle"+'\n'+
                            '/help- Send me an image about (AirPlane, Bicycle, Car or Ship)')
    

@mybot.message_handler(commands=['Recognition'])
def send_game(message):
    msg = mybot.reply_to(message,"Send me photo")
    mybot.register_next_step_handler(msg,recognize_vehicle)

def recognize_vehicle(message):
    model = tf.keras.models.load_model('Classification.h5')

    fileID = message.photo[-1].file_id
    file_info = mybot.get_file(fileID)
    downloaded_file = mybot.download_file(file_info.file_path)
    width = height = 224


    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    img = cv2.imread("image.jpg")
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(image ,(width,height))
    img = img / 255
    img = img.reshape(1,width,height,3) 

    result = np.argmax(model.predict(img))
    if result == 0:
        mybot.send_message(message.chat.id,'It must be airplane✈️')

    elif result == 1:
        mybot.send_message(message.chat.id,'It must be bicycle 🚲')

    elif result == 2:
        mybot.send_message(message.chat.id,'It must be car🚘')

    elif result == 3:
        mybot.send_message(message.chat.id,'It must be ship🛳')

    
mybot.enable_save_next_step_handlers(delay=2)
mybot.load_next_step_handlers()

mybot.polling()
