import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

from glob import glob
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tensorflow.keras.datasets import mnist

class EnsembleClassifier:

    model_names = ['angel_kanji_model', 'john_kanji_model', 'justin_kanji_model']
    model_name_pattern = '*model*'
    trained_models = {}
    missclassifications = {}
    y_true = []
    y_pred = []


    def __init__(self, dir) -> None:
        working_dir = os.path.join(os.getcwd(), dir)

        for name in glob(os.path.join(working_dir, self.model_name_pattern)):
            model_dir = os.path.join(working_dir, name)
            try:
                self.trained_models[name] = models.load_model(model_dir)
                print(f'LOADING MODEL: {name}\n')
                self.trained_models[name].summary()
                print('')
            except Exception as e:
                print(e)


    def predict(self, element):
        model_predictions = [None] * len(self.trained_models.keys())

        # Use each individual model to predict
        for i, (name, model) in enumerate(self.trained_models.items()):
            model_predictions[i] = model.predict(element)

        model_predictions = np.sum(np.array(model_predictions), axis=0)

        return np.argmax(model_predictions)

    def mnist_predict(self, element):
        model_predictions = [None] * len(self.trained_models.keys())

        # Use each individual model to predict
        for i, (name, model) in enumerate(self.trained_models.items()):
            model_predictions[i] = model.predict(element.reshape(1, 28, 28))

        model_predictions = np.sum(np.array(model_predictions), axis=0)

        return np.argmax(model_predictions)
        

    def validate(self, validation_data):
        for i, (element, label) in tqdm(enumerate(validation_data), ncols=100, desc='Validation Progress'):
            true_label = validation_data.class_names[label[0].numpy()]
            prediction = self.predict(element)
            pred_label = validation_data.class_names[prediction]

            self.y_true.append(true_label)
            self.y_pred.append(pred_label)

            if pred_label != true_label:
                # print(f'Model confused: {pred_label} for {true_label}')

                if pred_label in self.missclassifications.keys():
                    self.missclassifications[pred_label] += 1
                else:
                    self.missclassifications[pred_label] = 1
        
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(conf_matrix, 
                                index=[i for i in validation_data.class_names], 
                                columns=[i for i in validation_data.class_names])

        plt.figure(figsize=(40,40))
        ax = sns.heatmap(df_cm, annot=True, vmax=8)
        ax.set(xlabel="Predicted", ylabel="True", title=f'Ensemble Model Confusion Matrix for: {len(validation_data.class_names)} classes')
        ax.xaxis.tick_top()
        plt.xticks(rotation=90)
        plt.show()
        print('')


    def validate_mnist(self, x, y):
        for i, (element) in tqdm(enumerate(x), ncols=100, desc='Validation Progress'):
            true_label = y[i]
            pred_label = self.mnist_predict(element)
            
            self.y_true.append(true_label)
            self.y_pred.append(pred_label)

            if pred_label != true_label:
                # print(f'Model confused: {pred_label} for {true_label}')

                if pred_label in self.missclassifications.keys():
                    self.missclassifications[pred_label] += 1
                else:
                    self.missclassifications[pred_label] = 1

        labels = '0 1 2 3 4 5 6 7 8 9'.split(' ')
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(conf_matrix, 
                                index=[i for i in labels], 
                                columns=[i for i in labels])

        plt.figure(figsize=(40,40))
        ax = sns.heatmap(df_cm, annot=True, vmax=8)
        ax.set(xlabel="Predicted", ylabel="True", title=f'Ensemble Model Confusion Matrix for: {len(labels)} classes')
        ax.xaxis.tick_top()
        plt.xticks(rotation=90)
        plt.show()
        print('')
            

    def demo(self, validation_data):
        for i, (element, label) in enumerate(validation_data):
            true_label = validation_data.class_names[label[0].numpy()]
            prediction = self.predict(element)
            pred_label = validation_data.class_names[prediction]
            print(f'PRED: {pred_label}')
            print(f'TRUE: {true_label}')
            if pred_label != true_label:
                print(f'Model confused: {pred_label} for {true_label}')

            self.plot_image(element.numpy()[0].astype("uint8"), true_label)

            if i >= 10:
                break
    

    def plot_image(self, image, label):
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(label)
        plt.axis("off")
        plt.show()
        

    def plot_missclassifications(self):
        plt.bar(self.missclassifications.keys(), 
                self.missclassifications.values(), 
                1.0, color='r')
        plt.show()



def run_mnist_experiments():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    my_ensemble_model = EnsembleClassifier('./mnist_models')
    my_ensemble_model.validate_mnist(test_images, test_labels)



def run_kkanji_experiments():
    new_kkanji_class_final_dataset_val = tf.keras.utils.image_dataset_from_directory(
                                        './Code/datasets/class_final_dataset',
                                        validation_split=0.3,
                                        subset="validation",
                                        seed=132,
                                        image_size=(64, 64),
                                        color_mode = "grayscale",
                                        batch_size=1)
    
    my_ensemble_model = EnsembleClassifier('./Code/testing_models')
    my_ensemble_model.demo(new_kkanji_class_final_dataset_val)
    # my_ensemble_model.validate(new_kkanji_class_final_dataset_val)


if __name__ == "__main__":    
    run_mnist_experiments()
    # run_kkanji_experiments()
    # 96.13% accuracy on validation data
