# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** [Your Name Here]  
- **Register Number:** [Your Register Number Here]

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python

# Importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper function to display image using matplotlib
def display(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

img_gray = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)

```

#### 2. Print the image width, height & Channel.
```python
print(f"Width: {img_gray.shape[1]}, Height: {img_gray.shape[0]}, Channels: 1 (Grayscale)")
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale Eagle")
plt.axis('off')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight_Gray.png', img_gray)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img_color = cv2.imread('Eagle_in_Flight_Gray.png')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
print(f"Width: {img_color.shape[1]}, Height: {img_color.shape[0]}, Channels: {img_color.shape[2]}")
display("Color Image", img_color)
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
img = cv2.imread('Eagle_in_Flight.jpg')
eagle_crop = img[50:300, 100:400]  # Adjust based on actual image
display("Cropped Eagle", eagle_crop)
```

#### 8. Resize the image up by a factor of 2x.
```python
eagle_resized = cv2.resize(eagle_crop, None, fx=2, fy=2)
display("Resized Eagle", eagle_resized)
```

#### 9. Flip the cropped/resized image horizontally.
```python
eagle_flipped = cv2.flip(eagle_resized, 1)
display("Flipped Eagle", eagle_flipped)
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
apollo_img = cv2.imread('Apollo-11-launch.jpg')
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
text_size = cv2.getTextSize(text, font_face, 2, 2)[0]
text_x = (apollo_img.shape[1] - text_size[0]) // 2
text_y = apollo_img.shape[0] - 20
cv2.putText(apollo_img, text, (text_x, text_y), font_face, 2, (255, 255, 255), 2)
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = (255, 0, 255)  
cv2.rectangle(apollo_img, (100, 50), (400, 700), rect_color, 3)  
```

#### 13. Display the final annotated image.
```python
display("Apollo 11 Annotated", apollo_img)
```

#### 14. Read the image ('Boy.jpg').
```python
img_boy = cv2.imread('Boy.jpg')
```

#### 15. Adjust the brightness of the image.
```python
matrix = np.ones(img_boy.shape, dtype="uint8") * 50  # Brightness change
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img_boy, matrix)
img_darker = cv2.subtract(img_boy, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
display("Original", img_boy)
display("Darker", img_darker)
display("Brighter", img_brighter)
```

#### 18. Modify the image contrast.
```python
matrix1 = cv2.convertScaleAbs(img_boy, alpha=1.1, beta=0)
matrix2 = cv2.convertScaleAbs(img_boy, alpha=1.2, beta=0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
display("Contrast 1.1", matrix1)
display("Contrast 1.2", matrix2)
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
B, G, R = cv2.split(img_boy)
plt.figure(figsize=(10,3))
plt.subplot(1,3,1), plt.imshow(B, cmap='gray'), plt.title("Blue Channel"), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(G, cmap='gray'), plt.title("Green Channel"), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(R, cmap='gray'), plt.title("Red Channel"), plt.axis('off')
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
img_merged = cv2.merge((B, G, R))
display("Merged BGR", img_merged)
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv_img = cv2.cvtColor(img_boy, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_img)
plt.figure(figsize=(10,3))
plt.subplot(1,3,1), plt.imshow(H, cmap='gray'), plt.title("Hue"), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(S, cmap='gray'), plt.title("Saturation"), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(V, cmap='gray'), plt.title("Value"), plt.axis('off')
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge((H, S, V))
merged_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
display("Merged HSV to BGR", merged_bgr)
```

## Output:
- **i)** Read and Display an Image.
-   
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

