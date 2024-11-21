import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import os
import threading

# Load and preprocess the image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (512, 512))
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Load VGG model
def load_vgg_model():
    vgg = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
    return vgg

# Build the style transfer model
def build_style_transfer_model(vgg_model):
    content_layers = ['block5_conv2']  # Content layer
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Style layers

    outputs = [vgg_model.get_layer(name).output for name in content_layers + style_layers]
    
    return tf.keras.Model(inputs=vgg_model.input, outputs=outputs)

# Define the style transfer function
def style_transfer(content_image, style_image, model, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    generated_image = tf.Variable(content_image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    content_output = model(content_image)
    style_output = model(style_image)

    content_features = content_output[0]
    style_features = style_output[1:]

    style_targets = [gram_matrix(style_feature) for style_feature in style_features]

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            generated_output = model(generated_image)
            generated_content_features = generated_output[0]
            generated_style_features = generated_output[1:]

            content_loss = tf.reduce_mean(tf.square(generated_content_features - content_features))

            style_loss = 0
            for generated_style_feature, style_target in zip(generated_style_features, style_targets):
                style_loss += tf.reduce_mean(tf.square(gram_matrix(generated_style_feature) - style_target))

            total_loss = content_weight * content_loss + style_weight * style_loss

        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {total_loss.numpy()}")

    return generated_image

def gram_matrix(tensor):
    batch_size, height, width, channels = tensor.shape
    tensor = tf.reshape(tensor, (height * width, channels))
    gram = tf.matmul(tf.transpose(tensor), tensor)
    return gram / tf.cast(height * width, tf.float32)

class StyleTransferApp:
    def __init__(self, master):
        self.master = master
        master.title("Style Transfer App")

        self.dataset_directory = "D:/SLIIT/Y3S1/DIP/dataset"

        self.label = Label(master, text="Upload Content and Style Images for Style Transfer")
        self.label.pack()

        self.upload_content_button = Button(master, text="Upload Content Image", command=self.upload_content_image)
        self.upload_content_button.pack()

        self.upload_style_button = Button(master, text="Upload Style Image", command=self.upload_style_image)
        self.upload_style_button.pack()

        self.status_label = Label(master, text="")
        self.status_label.pack()

        self.style_transfer_button = Button(master, text="Start Style Transfer", command=self.start_style_transfer_thread)
        self.style_transfer_button.pack()

        self.content_image_label = Label(master)
        self.style_image_label = Label(master)
        self.stylized_image_label = Label(master)

        self.content_image_path = None
        self.style_image_path = None

    def upload_content_image(self):
        initial_dir = os.path.join(self.dataset_directory, "content_images")
        self.content_image_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select Content Image")
        self.status_label.config(text=f"Selected Content Image: {self.content_image_path}")
        self.display_image(self.content_image_path, self.content_image_label)

    def upload_style_image(self):
        initial_dir = os.path.join(self.dataset_directory, "style_images")
        self.style_image_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select Style Image")
        self.status_label.config(text=f"Selected Style Image: {self.style_image_path}")
        self.display_image(self.style_image_path, self.style_image_label)

    def display_image(self, image_path, label):
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
        label.pack()

    def start_style_transfer_thread(self):
        transfer_thread = threading.Thread(target=self.run_style_transfer)
        transfer_thread.start()

    def run_style_transfer(self):
        if self.content_image_path and self.style_image_path:
            content_image = load_image(self.content_image_path)
            style_image = load_image(self.style_image_path)

            vgg_model = load_vgg_model()
            model = build_style_transfer_model(vgg_model)

            self.status_label.config(text="Performing Style Transfer...")
            
            # Run the style transfer
            stylized_image = style_transfer(content_image, style_image, model)

            # Schedule result display back on the main thread using `after()`:
            self.master.after(0, self.show_result, stylized_image)
            self.master.after(0, self.status_label.config, {"text": "Style Transfer Completed"})
        else:
            self.status_label.config(text="Please upload both content and style images!")

    def show_result(self, stylized_image):
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = stylized_image.numpy()
        stylized_image = np.clip(stylized_image * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(stylized_image)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)

        self.stylized_image_label.config(image=img)
        self.stylized_image_label.image = img
        self.stylized_image_label.pack()

if __name__ == "__main__":
    root = Tk()
    my_gui = StyleTransferApp(root)
    root.mainloop()
