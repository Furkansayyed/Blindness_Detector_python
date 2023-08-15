import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = cv2.resize(image, (100, 100))  # Resize to a common size
    return processed_image.flatten()

# Load and preprocess dataset
normal_images = [preprocess_image("normal_right_eye.jpg"), preprocess_image("normal_left_eye.jpg")]
abnormal_images = [preprocess_image("abnormal_right_eye.jpg"), preprocess_image("abnormal_left_eye.jpg")]
labels = [0] * len(normal_images) + [1] * len(abnormal_images)

# Train a Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(np.vstack((normal_images, abnormal_images)), labels)

# Function to handle image upload
def upload_image():
    result_label.config(text=" ")
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        # image = image.resize((200, 200), Image.LANCZOS)
        image = image.resize((200, 200))
        img = ImageTk.PhotoImage(image)
        img_label.config(image=img)
        img_label.image = img
        predict_button.config(state=tk.NORMAL)
        app.file_path = file_path

# Function to perform prediction
def predict_blindness():
    if hasattr(app, 'file_path'):
        processed_image = preprocess_image(app.file_path)
        prediction = svm_model.predict([processed_image])
        result_label.config(text="The Uploaded Fundus eye is Normal" if prediction == 0 else "The Uploaded Fundus eye is Abnormal")

# Create the main UI window
app = tk.Tk()
app.title("Blindness Detection App")
app.geometry("500x500")
app.minsize(500,500)
app.maxsize(600,600)

# UI components

bigText =tk. Label(text='Image to Pencil Sketch App ', font='Helvetica 20 bold italic', foreground='crimson').place(x=65, y=30)

upload_button = tk.Button(app, text="Upload Image", command=upload_image, font='Helvetica 12 bold', fg='blue')
upload_button.place(x=185, y=80)

img_label = tk.Label(app)
img_label.place(x=150, y=130)

predict_button = tk.Button(app, text="Predict", command=predict_blindness, state=tk.DISABLED,  font='Helvetica 12 bold',fg='red')
predict_button.place(x=210, y=350)

result_label = tk.Label(app, text="", font="Arial 16 bold", fg='limegreen')
result_label.place(x=76, y=400)

extBtn = tk.Button(app, text="Exit", command=app.destroy, font='Helvetica 10 bold',bg='red', fg='white')
extBtn.place(x=450, y=450)

app.mainloop()
