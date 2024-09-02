import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Global variables to store images
img = None
img_display = None
processed_image = None

# Function to upload an image
def upload_image():
    global img, img_display, processed_image
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
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

# Function to rotate the image
def rotate_image():
    global processed_image
    if processed_image is not None:
        angle = int(angle_entry.get()) % 360
        (h, w) = processed_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed_image = cv2.warpAffine(processed_image, M, (w, h))
        display_images()

# Function to crop the image
def crop_image():
    global processed_image
    if processed_image is not None:
        x_start = int(crop_x_entry.get())
        y_start = int(crop_y_entry.get())
        width = int(crop_width_entry.get())
        height = int(crop_height_entry.get())
        processed_image = processed_image[y_start:y_start+height, x_start:x_start+width]
        display_images()

# Function to flip the image (invert colors)
def flip_image():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.bitwise_not(processed_image)
        display_images()

# Function to save the processed image
def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, processed_image)
            messagebox.showinfo("Image Saved", f"Image has been saved as {file_path}")

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

# Advanced Filters

# Sharpening filter
def apply_sharpen():
    global processed_image
    if processed_image is not None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv2.filter2D(src=processed_image, ddepth=-1, kernel=kernel)
        display_images()

# Smoothing filter
def apply_smooth():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.GaussianBlur(processed_image, (15, 15), 0)
        display_images()

# Edge detection
def apply_edge_detection():
    global processed_image
    if processed_image is not None:
        processed_image = cv2.Canny(processed_image, 100, 200)
        display_images()

# Embossing filter
def apply_emboss():
    global processed_image
    if processed_image is not None:
        kernel = np.array([[ -2, -1, 0],
                           [ -1,  1, 1],
                           [  0,  1, 2]])
        processed_image = cv2.filter2D(src=processed_image, ddepth=-1, kernel=kernel)
        display_images()

# Intensity Manipulation

# Increase contrast
def increase_contrast():
    global processed_image
    if processed_image is not None:
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        processed_image = cv2.merge((l, a, b))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)
        display_images()

# Adjust color balance
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

# Region-based segmentation
def apply_region_based_segmentation():
    global processed_image
    if processed_image is not None:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(processed_image, markers)
        processed_image[markers == -1] = [255, 0, 0]
        
        display_images()

# Initialize the Tkinter window
root = tk.Tk()
root.title("Image Processing Application")

# Image Upload Button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.grid(row=0, column=0, padx=10, pady=10)

# Color Change Options
color_frame = tk.LabelFrame(root, text="Color Change", padx=10, pady=10)
color_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

color_options = ["Color", "BW", "Grayscale"]
for mode in color_options:
    color_btn = tk.Button(color_frame, text=mode, command=lambda m=mode: change_color(m))
    color_btn.pack(side="left", padx=5, pady=5)

# Rotation
rotate_frame = tk.LabelFrame(root, text="Rotate Image", padx=10, pady=10)
rotate_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

angle_label = tk.Label(rotate_frame, text="Angle:")
angle_label.pack(side="left")

angle_entry = tk.Entry(rotate_frame, width=5)
angle_entry.insert(0, "90")
angle_entry.pack(side="left", padx=5)

rotate_btn = tk.Button(rotate_frame, text="Rotate", command=rotate_image)
rotate_btn.pack(side="left", padx=5, pady=5)

# Cropping
crop_frame = tk.LabelFrame(root, text="Crop Image", padx=10, pady=10)
crop_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

crop_x_label = tk.Label(crop_frame, text="X:")
crop_x_label.pack(side="left")

crop_x_entry = tk.Entry(crop_frame, width=5)
crop_x_entry.insert(0, "50")
crop_x_entry.pack(side="left", padx=5)

crop_y_label = tk.Label(crop_frame, text="Y:")
crop_y_label.pack(side="left")

crop_y_entry = tk.Entry(crop_frame, width=5)
crop_y_entry.insert(0, "50")
crop_y_entry.pack(side="left", padx=5)

crop_width_label = tk.Label(crop_frame, text="Width:")
crop_width_label.pack(side="left")

crop_width_entry = tk.Entry(crop_frame, width=5)
crop_width_entry.insert(0, "100")
crop_width_entry.pack(side="left", padx=5)

crop_height_label = tk.Label(crop_frame, text="Height:")
crop_height_label.pack(side="left")

crop_height_entry = tk.Entry(crop_frame, width=5)
crop_height_entry.insert(0, "100")
crop_height_entry.pack(side="left", padx=5)

crop_btn = tk.Button(crop_frame, text="Crop", command=crop_image)
crop_btn.pack(side="left", padx=5, pady=5)

reset_crop_btn = tk.Button(crop_frame, text="Reset Crop Size", command=reset_crop_size)
reset_crop_btn.pack(side="left", padx=5, pady=5)

# Flip Image Button
flip_btn = tk.Button(root, text="Flip Image", command=flip_image)
flip_btn.grid(row=4, column=0, padx=10, pady=10)

# Save Image Button
save_btn = tk.Button(root, text="Save Image", command=save_image)
save_btn.grid(row=5, column=0, padx=10, pady=10)

# Filters Frame
filters_frame = tk.LabelFrame(root, text="Filters", padx=10, pady=10)
filters_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

sharpen_btn = tk.Button(filters_frame, text="Sharpen", command=apply_sharpen)
sharpen_btn.pack(side="left", padx=5, pady=5)

smooth_btn = tk.Button(filters_frame, text="Smooth", command=apply_smooth)
smooth_btn.pack(side="left", padx=5, pady=5)

edge_detection_btn = tk.Button(filters_frame, text="Edge Detection", command=apply_edge_detection)
edge_detection_btn.pack(side="left", padx=5, pady=5)

emboss_btn = tk.Button(filters_frame, text="Emboss", command=apply_emboss)
emboss_btn.pack(side="left", padx=5, pady=5)

# Intensity Manipulation Frame
intensity_frame = tk.LabelFrame(root, text="Intensity Manipulation", padx=10, pady=10)
intensity_frame.grid(row=7, column=0, padx=10, pady=10, sticky="ew")

contrast_btn = tk.Button(intensity_frame, text="Increase Contrast", command=increase_contrast)
contrast_btn.pack(side="left", padx=5, pady=5)

color_balance_btn = tk.Button(intensity_frame, text="Color Balance", command=adjust_color_balance)
color_balance_btn.pack(side="left", padx=5, pady=5)

# Segmentation Frame
segmentation_frame = tk.LabelFrame(root, text="Segmentation", padx=10, pady=10)
segmentation_frame.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

segmentation_btn = tk.Button(segmentation_frame, text="Region-based Segmentation", command=apply_region_based_segmentation)
segmentation_btn.pack(side="left", padx=5, pady=5)

# Image Display Area
original_label = tk.Label(root)
original_label.grid(row=0, column=1, rowspan=9, padx=10, pady=10)

processed_label = tk.Label(root)
processed_label.grid(row=0, column=2, rowspan=9, padx=10, pady=10)

root.mainloop()
