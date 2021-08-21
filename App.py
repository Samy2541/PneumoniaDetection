import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from tkinter import filedialog,Text,Label
from tkinter import *
import tkinter


imagepath = ""
window = tkinter.Tk()
window.title("Pneumonia Detection")
window.iconbitmap("medica.ico")
window.geometry("862x519")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 519,
    width = 862,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = "background.png")
background = canvas.create_image(
    374.0, 295.5,
    image=background_img)

canvas.create_text(
    220.0, 78.5,
    text = "Pneumonia detection",
    fill = "#ffffff",
    font = ("Bite Chocolate", int(30.0)))

canvas.create_text(
    640.0, 120,
    text = "Please consider uploading"
           " X-ray images only to insure"
           " accurate results.",
    fill = "#5ea9a5",
    font = ("Bite Chocolate", int(10.0)))


def upload():
    global imagepath
    imagepath = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("jpeg files", "*.jpeg"), ("all files", "*.*")))



def predict():
    model = keras.models.load_model('KerasSaved')
    classes = ["Normal", "Pneumonia"]
    img = image.load_img(imagepath, target_size=(196, 196))
    plt.imshow(img)
    img = image.img_to_array(img)
    img = preprocess_input(img)
    test_image = img.reshape(1, 196, 196, 3)
    result = model.predict(test_image)
    output = np.argmax(result)
    plt.text(80, -10, classes[output], fontsize=18, bbox=dict(fill=False, linewidth=2))
    plt.show()

b0 = tkinter.Button(bg="#5ea9a5",fg="white",text="Upload",font='sans 16 bold', borderwidth=0, highlightthickness=0, command=upload, relief="flat")

b0.place(
    x = 572, y = 150,
    width = 152,
    height = 51)

b1 = tkinter.Button(bg="#5ea9a5",fg="white",text="Diagnose",font='sans 16 bold', borderwidth=0, highlightthickness=0, command=predict, relief="flat")

b1.place(
    x = 572, y = 300,
    width = 152,
    height = 51)

window.resizable(False, False)
window.mainloop()