import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

def predict_img_result():
    ## 修改图片大小为能输入模型的大小
    to_predict_name = 'images/tailuo/002.jpg'
    img_name = cv2.imread(to_predict_name)
    img_init = cv2.resize(img_name, (224, 224))  
    if not img_init:
        print('图片存在问题，读取失败，请更换图片进行预测！！！')
    cv2.imwrite('images/target.png', img_init)
    ##　正式开始预测
    class_names = ['迪迦奥特曼', '捷克奥特曼', '赛文埃特曼', '泰罗奥特曼']
    model = tf.keras.models.load_model("models/cnn_aoteman.h5")  
    img = Image.open('images/target.png')
    img = np.asarray(img) 
    outputs = model.predict(img.reshape(1, 224, 224, 3))  
    result_index = int(np.argmax(outputs))
    result = class_names[result_index]  
    print(result) 

if __name__ == '__main__':
    predict_img_result()