import tkinter as tk    
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy
from tkinter import Tk, Button
import subprocess
import os
from style_transfer_model import load_image, load_vgg_model, build_style_transfer_model, style_transfer

# Global variables to store images and history for undo functionality
img = None
img_display = None
processed_image = None
undo_stack = []
start_x, start_y = None, None
rect_id = None

# Function to upload an image
def upload_image():
    global img, img_display, processed_image
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (500, 500))  # Resize image to fit the frame
        img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        processed_image = img.copy()
        display_images()

# Function to display the original and processed images
def display_images():
    global img_display, processed_image
    original_label.config(image=img_display)
    original_label.image = img_display

    if processed_image is not None:
        processed_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)))
        processed_label.config(image=processed_display)
        processed_label.image = processed_display

# Function to change the color of the image
def change_color(mode):
    global processed_image
    if img is not None:
        if mode == "Color":
            processed_image = img.copy()
        elif mode == "BW":
            _, processed_image = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        elif mode == "Grayscale":
            processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        display_images()

# Function to rotate the image clockwise
def rotate_image():
    global processed_image
    if processed_image is not None:
        try:
            angle = int(angle_entry.get()) % 360
            angle = 360 - angle  # Subtract from 360 to rotate clockwise
            (h, w) = processed_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed_image = cv2.warpAffine(processed_image, M, (w, h))
            display_images()
        except ValueError:
            messagebox.showerror("Error", "Invalid angle value")

# Function to crop the image smoothly
def crop_image():
    global processed_image
    if processed_image is not None:
        try:
            x_start = int(crop_x_entry.get())
            y_start = int(crop_y_entry.get())
            width = int(crop_width_entry.get())
            height = int(crop_height_entry.get())

            # Ensure the crop coordinates are within the image dimensions
            (h, w) = processed_image.shape[:2]
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(x_start + width, w)
            y_end = min(y_start + height, h)
            
            # Save the current state before cropping for undo functionality
            save_state_for_undo()
            
            # Crop and resize the image smoothly
            processed_image = processed_image[y_start:y_end, x_start:x_end]
            processed_image = cv2.resize(processed_image, (w, h), interpolation=cv2.INTER_LINEAR)
            display_images()
        except ValueError:
            messagebox.showerror("Error", "Invalid crop parameters")

# Function to reset crop size
def reset_crop_size():
    crop_x_entry.delete(0, tk.END)
    crop_x_entry.insert(0, "50")
    crop_y_entry.delete(0, tk.END)
    crop_y_entry.insert(0, "50")
    crop_width_entry.delete(0, tk.END)
    crop_width_entry.insert(0, "100")
    crop_height_entry.delete(0, tk.END)
    crop_height_entry.insert(0, "100")


    # Mouse event functions for cropping
def start_crop(event):
    global start_x, start_y, rect_id
    if processed_image is not None:
        start_x = event.x
        start_y = event.y
        if rect_id:
            processed_label.delete(rect_id)

def draw_crop_rectangle(event):
    global rect_id, start_x, start_y, processed_image
    if processed_image is not None and start_x is not None and start_y is not None:
        # Ensure the coordinates are within the image bounds
        end_x, end_y = min(event.x, processed_label.winfo_width()), min(event.y, processed_label.winfo_height())
        
        # Create a copy of the original image for drawing
        img_copy = processed_image.copy()

        # Draw the rectangle on the copy
        cv2.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # Convert the image to display it in the Tkinter label
        img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy_pil = Image.fromarray(img_copy_rgb)
        img_copy_tk = ImageTk.PhotoImage(img_copy_pil)

        # Update the label with the new image containing the rectangle
        processed_label.config(image=img_copy_tk)
        processed_label.image = img_copy_tk

def end_crop(event):
    global start_x, start_y, end_x, end_y, processed_image
    if processed_image is not None and start_x and start_y:
        end_x, end_y = event.x, event.y
        # Ensure coordinates are within image bounds
        img_height, img_width = processed_image.shape[:2]
        x1, y1 = max(0, min(start_x, end_x)), max(0, min(start_y, end_y))
        x2, y2 = min(img_width, max(start_x, end_x)), min(img_height, max(start_y, end_y))
        if x2 - x1 > 0 and y2 - y1 > 0:
            save_state_for_undo()
            processed_image = processed_image[y1:y2, x1:x2]
            display_images()
        else:
            messagebox.showwarning("Crop Error", "Invalid crop area selected.")
        start_x = start_y = end_x = end_y = None

# Function to delete the image
def delete_image():
    global img, processed_image, img_display
    img = None
    processed_image = None
    img_display = None
    original_label.config(image=None)
    processed_label.config(image=None)

# Function to flip the image (invert colors)
def flip_image_horizontal():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.flip(processed_image, 1)  # 1 for horizontal flip
        display_images()

def flip_image_vertical():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.flip(processed_image, 0)  # 0 for vertical flip
        display_images()

# Function to save the processed image
def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", ".jpg"), ("PNG files", ".png"), ("All files", ".")])
        if file_path:
            cv2.imwrite(file_path, processed_image)
            messagebox.showinfo("Image Saved", f"Image has been saved as {file_path}")

# Function to save the current state for undo functionality
def save_state_for_undo():
    global undo_stack, processed_image
    if processed_image is not None:
        # Save a deep copy of the current state
        undo_stack.append(copy.deepcopy(processed_image))


# Function to undo the last operation
def undo():
    global processed_image, undo_stack
    if undo_stack:
        # Revert to the last state
        processed_image = undo_stack.pop()
        # Update the display to show the reverted image
        display_images()
    else:
        # Inform the user if there's nothing to undo
        messagebox.showwarning("Undo Error", "No actions to undo.")


def crop_image():
    global processed_image
    if processed_image is not None:
        try:
            x_start = int(crop_x_entry.get())
            y_start = int(crop_y_entry.get())
            width = int(crop_width_entry.get())
            height = int(crop_height_entry.get())

            # Ensure the crop coordinates are within the image dimensions
            (h, w) = processed_image.shape[:2]
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(x_start + width, w)
            y_end = min(y_start + height, h)
            
            # Save the current state before cropping for undo functionality
            save_state_for_undo()
            
            # Crop and resize the image smoothly
            processed_image = processed_image[y_start:y_end, x_start:x_end]
            processed_image = cv2.resize(processed_image, (w, h), interpolation=cv2.INTER_LINEAR)
            display_images()
        except ValueError:
            messagebox.showerror("Error", "Invalid crop parameters")

# Function to reset crop size
def reset_crop_size():
    crop_x_entry.delete(0, tk.END)
    crop_x_entry.insert(0, "50")
    crop_y_entry.delete(0, tk.END)
    crop_y_entry.insert(0, "50")
    crop_width_entry.delete(0, tk.END)
    crop_width_entry.insert(0, "100")
    crop_height_entry.delete(0, tk.END)
    crop_height_entry.insert(0, "100")


    # Mouse event functions for cropping
def start_crop(event):
    global start_x, start_y, rect_id
    if processed_image is not None:
        start_x = event.x
        start_y = event.y
        if rect_id:
            processed_label.delete(rect_id)

def draw_crop_rectangle(event):
    global rect_id, start_x, start_y, processed_image
    if processed_image is not None and start_x is not None and start_y is not None:
        # Ensure the coordinates are within the image bounds
        end_x, end_y = min(event.x, processed_label.winfo_width()), min(event.y, processed_label.winfo_height())
        
        # Create a copy of the original image for drawing
        img_copy = processed_image.copy()

        # Draw the rectangle on the copy
        cv2.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # Convert the image to display it in the Tkinter label
        img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy_pil = Image.fromarray(img_copy_rgb)
        img_copy_tk = ImageTk.PhotoImage(img_copy_pil)

        # Update the label with the new image containing the rectangle
        processed_label.config(image=img_copy_tk)
        processed_label.image = img_copy_tk

def end_crop(event):
    global start_x, start_y, end_x, end_y, processed_image
    if processed_image is not None and start_x and start_y:
        end_x, end_y = event.x, event.y
        # Ensure coordinates are within image bounds
        img_height, img_width = processed_image.shape[:2]
        x1, y1 = max(0, min(start_x, end_x)), max(0, min(start_y, end_y))
        x2, y2 = min(img_width, max(start_x, end_x)), min(img_height, max(start_y, end_y))
        if x2 - x1 > 0 and y2 - y1 > 0:
            save_state_for_undo()
            processed_image = processed_image[y1:y2, x1:x2]
            display_images()
        else:
            messagebox.showwarning("Crop Error", "Invalid crop area selected.")
        start_x = start_y = end_x = end_y = None


# Function to flip the image (invert colors)

def flip_image_horizontal():
    global processed_image
    if processed_image is not None:
        save_state_for_undo()  # Save the current state for undo functionality
        processed_image = cv2.flip(processed_image, 1)  # 1 for horizontal flip
        display_images()

def flip_image_vertical():
    global processed_image
    if processed_image is not None:
        save_state_for_undo()  # Save the current state for undo functionality
        processed_image = cv2.flip(processed_image, 0)  # 0 for vertical flip
        display_images()


# Function to save the processed image
def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", ".jpg"), ("PNG files", ".png"), ("All files", ".")])
        if file_path:
            cv2.imwrite(file_path, processed_image)
            messagebox.showinfo("Image Saved", f"Image has been saved as {file_path}")

# Function to save the current state for undo functionality
def save_state_for_undo():
    global undo_stack, processed_image
    if processed_image is not None:
        # Save a deep copy of the current state
        undo_stack.append(copy.deepcopy(processed_image))

# Advanced Filters
def apply_sharpen():
    global processed_image
    if processed_image is not None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv2.filter2D(src=processed_image, ddepth=-1, kernel=kernel)
        display_images()

def apply_smooth():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.GaussianBlur(processed_image, (15, 15), 0)
        display_images()

def apply_edge_detection():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.Canny(processed_image, 100, 200)
        display_images()

def apply_emboss():
    global processed_image
    if processed_image is not None:
        kernel = np.array([[ -2, -1, 0],
                           [ -1,  1, 1],
                           [  0,  1, 2]])
        processed_image = cv2.filter2D(src=processed_image, ddepth=-1, kernel=kernel)
        display_images()

# Intensity Manipulation
def increase_contrast():
    global processed_image
    if processed_image is not None:
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        processed_image = cv2.merge((l, a, b))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)
        display_images()
# Adjust Hue
def adjust_hue(value):
    global processed_image
    if processed_image is not None:
        save_state_for_undo()
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.add(h, int(value))
        h = np.clip(h, 0, 179)
        processed_image = cv2.merge((h, s, v))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)
        display_images()

# Adjust Saturation
def adjust_saturation(value):
    global processed_image
    if processed_image is not None:
        save_state_for_undo()
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Convert value to float and scale for saturation adjustment
        adjustment_factor = 1 + (float(value) / 100.0)
        
        # Apply the saturation adjustment and ensure it stays within the valid range
        s = np.clip(s * adjustment_factor, 0, 255).astype(np.uint8)
        
        # Merge the channels back and convert to BGR color space
        adjusted_hsv = cv2.merge((h, s, v))
        processed_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        display_images()
def adjust_color_balance():
    global processed_image
    if processed_image is not None:
        r, g, b = cv2.split(processed_image)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        processed_image = cv2.merge((r, g, b))
        display_images()

# Image Segmentation
def apply_threshold_based_segmentation():
    global processed_image
    if processed_image is not None:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        display_images()

def apply_edge_based_segmentation():
    global processed_image
    if processed_image is not None:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        display_images()

def apply_morphological_methods_based_segmentation():
    global processed_image
    if processed_image is not None:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(processed_image, markers)
        processed_image[markers == -1] = [255, 0, 0]
        display_images()

def apply_region_based_segmentation():
    global processed_image
    if processed_image is not None:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(processed_image, markers)
        processed_image[markers == -1] = [255, 0, 0]
        display_images()



###########################################################################################

# Initialize the Tkinter window
root = tk.Tk()
root.title("Image Processing Application")

# Create the main layout
main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill='both')

# Create toolbar for main actions
toolbar = tk.Frame(main_frame, bd=1, relief=tk.RAISED, bg='#D3D3D3')
toolbar.pack(side=tk.TOP, fill=tk.X)

# Toolbar buttons centered
button_frame = tk.Frame(toolbar)
button_frame.pack(side=tk.TOP, padx=20)

upload_btn = tk.Button(button_frame, text="Upload Image", command=upload_image)
upload_btn.pack(side=tk.LEFT, padx=2, pady=2)

save_btn = tk.Button(button_frame, text="Save Image", command=save_image)
save_btn.pack(side=tk.LEFT, padx=2, pady=2)

delete_btn = tk.Button(button_frame, text="Delete Image", command=delete_image)
delete_btn.pack(side=tk.LEFT, padx=2, pady=2)

undo_btn = tk.Button(button_frame, text="Undo", command=undo)
undo_btn.pack(side=tk.LEFT, padx=2, pady=2)


# Create a frame for tools and filters
tools_frame = tk.LabelFrame(main_frame, text="Tools & Filters", padx=10, pady=10)
tools_frame.pack(side=tk.TOP, padx=10, pady=10, fill='x')

# Color Change Options
color_frame = tk.LabelFrame(tools_frame, text="Color Change", padx=10, pady=10)
color_frame.pack(side=tk.LEFT, padx=5, pady=5)

color_options = ["Color", "BW", "Grayscale"]
for mode in color_options:
    color_btn = tk.Button(color_frame, text=mode, command=lambda m=mode: change_color(m))
    color_btn.pack(side="left", padx=5, pady=5)

# Rotation
rotate_frame = tk.LabelFrame(tools_frame, text="Rotate Image", padx=10, pady=10)
rotate_frame.pack(side=tk.LEFT, padx=5, pady=5)

angle_label = tk.Label(rotate_frame, text="Angle:")
angle_label.pack(side="left")

angle_entry = tk.Entry(rotate_frame, width=5)
angle_entry.pack(side="left")

rotate_btn = tk.Button(rotate_frame, text="Rotate", command=rotate_image)
rotate_btn.pack(side="left", padx=5, pady=5)


# Cropping
crop_frame = tk.LabelFrame(tools_frame, text="Crop Image", padx=10, pady=10)
crop_frame.pack(side=tk.LEFT, padx=5, pady=5)

crop_x_label = tk.Label(crop_frame, text="X:")
crop_x_label.pack(side="left")

crop_x_entry = tk.Entry(crop_frame, width=5)
crop_x_entry.pack(side="left")

crop_y_label = tk.Label(crop_frame, text="Y:")
crop_y_label.pack(side="left")

crop_y_entry = tk.Entry(crop_frame, width=5)
crop_y_entry.pack(side="left")

crop_width_label = tk.Label(crop_frame, text="Width:")
crop_width_label.pack(side="left")

crop_width_entry = tk.Entry(crop_frame, width=5)
crop_width_entry.pack(side="left")

crop_height_label = tk.Label(crop_frame, text="Height:")
crop_height_label.pack(side="left")

crop_height_entry = tk.Entry(crop_frame, width=5)
crop_height_entry.pack(side="left")

crop_btn = tk.Button(crop_frame, text="Crop", command=crop_image)
crop_btn.pack(side="left", padx=5, pady=5)

reset_crop_btn = tk.Button(crop_frame, text="Reset Crop Size", command=reset_crop_size)
reset_crop_btn.pack(side="left", padx=5, pady=5)

# Advanced Filters
filter_frame = tk.LabelFrame(tools_frame, text="Advanced Filters", padx=10, pady=10)
filter_frame.pack(side=tk.LEFT, padx=5, pady=5)

sharpen_btn = tk.Button(filter_frame, text="Sharpen", command=apply_sharpen)
sharpen_btn.pack(side="left", padx=5, pady=5)

smooth_btn = tk.Button(filter_frame, text="Smooth", command=apply_smooth)
smooth_btn.pack(side="left", padx=5, pady=5)

edge_detection_btn = tk.Button(filter_frame, text="Edge Detection", command=apply_edge_detection)
edge_detection_btn.pack(side="left", padx=5, pady=5)

emboss_btn = tk.Button(filter_frame, text="Emboss", command=apply_emboss)
emboss_btn.pack(side="left", padx=5, pady=5)


# Frame for intensity, flip, and segmentation tools on the left
tool_controls_frame = tk.Frame(main_frame, width=200)
tool_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Intensity Manipulation Frame
intensity_frame = tk.LabelFrame(tool_controls_frame, text="Intensity Manipulation", padx=10, pady=10)
intensity_frame.pack(fill='x', pady=5)

contrast_btn = tk.Button(intensity_frame, text="Increase Contrast", command=increase_contrast)
contrast_btn.pack(side="top", padx=5, pady=5, fill='x')

color_balance_btn = tk.Button(intensity_frame, text="Adjust Color Balance", command=adjust_color_balance)
color_balance_btn.pack(side="top", padx=5, pady=5, fill='x')

# Image Flip Frame
flip_frame = tk.LabelFrame(tool_controls_frame, text="Image Flip", padx=10, pady=10)
flip_frame.pack(fill='x', pady=5)

flip_horizontal_btn = tk.Button(flip_frame, text="Flip Horizontal", command=flip_image_horizontal)
flip_horizontal_btn.pack(side="top", padx=5, pady=5, fill='x')

flip_vertical_btn = tk.Button(flip_frame, text="Flip Vertical", command=flip_image_vertical)
flip_vertical_btn.pack(side="top", padx=5, pady=5, fill='x')

# Image Segmentation Frame
segmentation_frame = tk.LabelFrame(tool_controls_frame, text="Image Segmentation", padx=10, pady=10)
segmentation_frame.pack(fill='x', pady=5)

threshold_btn = tk.Button(segmentation_frame, text="Threshold-Based", command=apply_threshold_based_segmentation)
threshold_btn.pack(side="top", padx=5, pady=5, fill='x')

edge_based_btn = tk.Button(segmentation_frame, text="Edge-Based", command=apply_edge_based_segmentation)
edge_based_btn.pack(side="top", padx=5, pady=5, fill='x')

morphological_btn = tk.Button(segmentation_frame, text="Morphological Methods", command=apply_morphological_methods_based_segmentation)
morphological_btn.pack(side="top", padx=5, pady=5, fill='x')

region_based_btn = tk.Button(segmentation_frame, text="Region-Based", command=apply_region_based_segmentation)
region_based_btn.pack(side="top", padx=5, pady=5, fill='x')

# Frame for original and processed images on the right
image_frame = tk.Frame(main_frame)
image_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=10, pady=10)

# Create frames for original and processed images
original_frame = tk.Frame(image_frame, width=500, height=500, bg='white')
original_frame.pack(side=tk.LEFT, expand=True, fill='both')

processed_frame = tk.Frame(image_frame, width=500, height=500, bg='white')
processed_frame.pack(side=tk.RIGHT, expand=True, fill='both')

# Labels for displaying images
original_label = tk.Label(original_frame)
original_label.pack(expand=True)

processed_label = tk.Label(processed_frame)
processed_label.pack(expand=True)

# Bind mouse events for cropping
processed_label.bind("<ButtonPress-1>", start_crop)
processed_label.bind("<B1-Motion>", draw_crop_rectangle)
processed_label.bind("<ButtonRelease-1>", end_crop)

# Load the content and style images
content_image = load_image("D:/SLIIT/Y3S1/DIP/dataset/content_images/content1.jpg")
style_image = load_image("D:/SLIIT/Y3S1/DIP/dataset/style_images/style1.jpg")

# Load the pre-trained model
model = create_vgg_model()  # Make sure the model-loading function exists in your style_transfer_model.py

# Perform the style transfer
stylized_image = style_transfer(content_image, style_image, model)

# Now you can display or save the stylized image

# Display the result (you might already have this part)
def show_images(content, style, output):
    plt.subplot(1, 3, 1)
    plt.title("Content Image")
    plt.imshow(content)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Style Image")
    plt.imshow(style)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Stylized Image")
    plt.imshow(output)
    plt.axis('off')

    plt.show()

# Load VGG19 model and build style transfer model
vgg = load_vgg_model()
model = build_style_transfer_model(vgg)

# Perform style transfer
stylized_image = style_transfer(content_image, style_image, model)

# Show the results
show_images(content_image[0].numpy(), style_image[0].numpy(), stylized_image[0].numpy())

def open_style_transfer_app():
    # Path to the style transfer script
    style_transfer_script = "D:/SLIIT/Y3S1/DIP/style_transfer_model.py"
    # Open the style transfer app
    subprocess.Popen(['python', style_transfer_script], cwd=os.path.dirname(style_transfer_script))

# Create the main application window
if __name__ == "__main__":
    root = Tk()
    root.title("DIP Project")

    # Button to open the style transfer app
    style_transfer_button = Button(root, text="Open Style Transfer App", command=open_style_transfer_app)
    style_transfer_button.pack(pady=20)


# Run the application
root.mainloop()