import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
from config import Congfig

def model(IMG_SHAPE=(224, 224, 3), class_num=12):
    # 加载预训练的 VGG16 模型，不包括顶层
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
    base_model.trainable = False

    # 创建模型的顶层
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),  # 可选的全连接层
        tf.keras.layers.Dropout(0.5),  # 可选的 Dropout 层，防止过拟合
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])

    model.summary()

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    Congfig.train(100, model, "train_path", "val_path","to_model_name")
    Congfig.test_model("train_path", "val_path", "model_path", "to_path")
