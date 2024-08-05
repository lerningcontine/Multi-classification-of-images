import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

class Congfig:
    def __init__(self) -> None:
        pass


    def data_load(self, data_dir, test_data_dir, img_height, img_width, batch_size):
        """
        加载数据集：
        这里主要是加载训练集和验证集
        """
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode='categorical',
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_dir,
            label_mode='categorical',
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names
    
    
    def show_loss_acc(self, history):
        """
        构建评估曲线：
        这里主要是准确率
        从history中提取模型训练集和验证集准确率信息和误差信息
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('results/results.png', dpi=100)

    def train(self, epochs, model, train_path:str, val_path:str, to_model_name:str):
        """
        训练函数：
        下面分别为验证集的路径和训练集的路径
        """
        begin_time = time()
        train_ds, val_ds, class_names = self.data_load(train_path,
                                                val_path, 224, 224, 16)
        print(class_names)
        model = model(class_num=len(class_names))
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        model.save(f"models/{to_model_name}.h5")
        end_time = time()
        run_time = end_time - begin_time
        print('该循环程序运行时间：', run_time, "s")  
        self.show_loss_acc(history)

    def test_model(self,train_path: str, val_path: str, model_path: str, to_path: str):
        """
        测试模型
        """
        test_ds, class_names = self.data_load(train_path,
                                                val_path, 224, 224, 16)
        model = tf.keras.models.load_model(model_path)
        test_real_labels = []
        test_pre_labels = []
        for test_batch_images, test_batch_labels in test_ds:
            test_batch_labels = test_batch_labels.numpy()
            test_batch_pres = model.predict(test_batch_images)

            test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
            test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
            for i in test_batch_labels_max:
                test_real_labels.append(i)

            for i in test_batch_pres_max:
                test_pre_labels.append(i)
        class_names_length = len(class_names)
        heat_maps = np.zeros((class_names_length, class_names_length))
        for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
            heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1
        heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
        heat_maps_float = heat_maps / heat_maps_sum
        self.show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                    save_name=to_path)
        
    
    def show_heatmaps(self, title, x_labels, y_labels, harvest, save_name):
        fig, ax = plt.subplots()
        im = ax.imshow(harvest, cmap="OrRd")
        ax.set_xticks(np.arange(len(y_labels)))
        ax.set_yticks(np.arange(len(x_labels)))
        ax.set_xticklabels(y_labels)
        ax.set_yticklabels(x_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, round(harvest[i, j], 2),
                            ha="center", va="center", color="black")
        ax.set_xlabel("Predict label")
        ax.set_ylabel("Actual label")
        ax.set_title(title)
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(save_name, dpi=100)

Config = Congfig()