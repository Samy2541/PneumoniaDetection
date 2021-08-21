import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

path = "C:/Users/DELL/PycharmProjects/pneumonia/chest_xray/"
# train directory
normal_dir = path+"NORMAL/"
pneu_dir = path+"PNEUMONIA/"

# variables for image size
img_width = 196
img_height = 196
batch_size = 64
epochs = 10

normal_cases = glob.glob(normal_dir + '*jpeg')
pneu_cases = glob.glob(pneu_dir + '*jpeg')

# Load Dataset(identify Image value and label)


def load_data(dir, type):
    values = []
    labels = []
    image_list = os.listdir(dir)
    for img_name in image_list:
        # Loading images
        img = image.load_img(dir + "/" + img_name, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        values.append(img)
        labels.append(type)
    return values, labels

normal_values, normal_labels = load_data(normal_dir, 0)
pneu_values, pneu_labels = load_data(pneu_dir, 1)
total_values = normal_values + pneu_values
total_labels = normal_labels + pneu_labels

total_values = np.array(total_values)
total_labels = to_categorical(total_labels)

input_layer = layers.Input(shape=(img_width, img_height, 3))
model_vgg16 = VGG16(weights='imagenet', input_tensor=input_layer, include_top=False)
last_layer = model_vgg16.output
# Add flatten layer: we are extending Neural Network by adding flatten layer
flatten = layers.Flatten()(last_layer)
output_layer = layers.Dense(2, activation='softmax')(flatten)
model = models.Model(inputs=input_layer, outputs=output_layer)
for layer in model.layers[:-1]:
    layer.trainable = False

train_values, test_values, train_labels, test_labels = train_test_split(total_values,total_labels, test_size=0.2, random_state=5)
test_values, val_values, test_labels, val_labels = train_test_split(test_values,test_labels,test_size=0.5,random_state=5)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(train_values,train_labels,epochs=epochs,batch_size=batch_size,verbose=True,validation_data=(val_values,val_labels))

json_file = model.to_json()
with open("SavedModel", "w") as file:
   file.write(json_file)
model.save_weights("weights.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs2 = range(len(acc))

plt.plot(epochs2, acc, 'r', label='Training accuracy')
plt.plot(epochs2, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

results = model.evaluate(test_values, test_labels, batch_size=batch_size)
print("test loss, test acc:", results)

model.save("KerasSaved")
print("Fitting the model completed.")

saved_model_dir = '/content/TFLite'
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)