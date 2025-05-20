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
- **Name:** SADHANA SHREE B  
- **Register Number:** 212223230177

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)
```

#### 2. Print the image width, height & Channel.
```python
image.shape
```

#### 3. Display the image using matplotlib imshow().
```python
img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray,cmap='grey')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png', image)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```pyhton
img = cv2.imread('Eagle_in_Flight.png')
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(color_img)
plt.title("Coloured Image")
plt.show()
img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
crop = color_img[0:450,200:550] 
plt.imshow(crop)
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape
```

#### 8. Resize the image up by a factor of 2x.
```python
res= cv2.resize(crop,(200*2, 200*2))
```

#### 9. Flip the cropped/resized image horizontally.
```python
flip= cv2.flip(res,1)
plt.imshow(flip)
plt.title("Flipped Horizontally")
plt.axis("off")
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img2=cv2.imread('Apollo-11-launch.jpg',cv2.IMREAD_COLOR)
img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_rgb2.shape
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = cv2.putText(img_rgb2, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
plt.imshow(text, cmap='gray')  
plt.title("Text Edit")
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rcol= (255, 0, 255)
cv2.rectangle(img_rgb2, (400, 50), (800, 650), rcol, 3) 
plt.imshow(img_rgb2)
plt.show()
```

#### 13. Display the final annotated image.
```python
plt.title("Annotated Image")
plt.imshow(img_rgb2)
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img_boy = cv2.imread('boy.jpg', cv2.IMREAD_COLOR)
img_rgbb= cv2.cvtColor(img_boy, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
m = np.ones(img_boy.shape, dtype="uint8") * 50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img_boy, m)  
img_darker = cv2.subtract(img_boy, m)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgbb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter[:,:,::-1]), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker[:,:,::-1]), plt.title("Darker Image"), plt.axis("off")
plt.show()
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(img_boy.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_boy.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img_boy.astype("float32"), matrix1).clip(0,255).astype("uint8")
img_higher2 = cv2.multiply(img_boy.astype("float32"), matrix2).clip(0,255).astype("uint8")
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_boy), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_higher1), plt.title("Higher Contrast (1.1x)"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_higher2), plt.title("Higher Contrast (1.2x)"), plt.axis("off")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(img_boy)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g, cmap='gray'), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r, cmap='gray'), plt.title("Red Channel"), plt.axis("off")
plt.show()
```

#### 21. Merge and display
```python
b, g, r = cv2.split(img_boy)
merged = cv2.merge([b, g, r])
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv_img = cv2.cvtColor(img_boy, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h, cmap='gray'), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s, cmap='gray'), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v, cmap='gray'), plt.title("Value Channel"), plt.axis("off")
plt.show()
```

#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
combined = np.concatenate((img_rgbb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()
```


## Output:
- **i)** Read and Display an Image.
  ![Screenshot 2025-05-20 200605](https://github.com/user-attachments/assets/6f6fa890-893f-4d66-8e92-767b9ccffb0e)
 ![Screenshot 2025-05-20 200612](https://github.com/user-attachments/assets/fcd3c701-2f0a-47f3-ae9d-00f12acdf31c)
![Screenshot 2025-05-20 200620](https://github.com/user-attachments/assets/dca72713-afe9-43e1-8e96-013138edc069)
![Screenshot 2025-05-20 200857](https://github.com/user-attachments/assets/0c8e331f-65f8-4be8-9047-fa0256113f0f)

![Screenshot 2025-05-20 200633](https://github.com/user-attachments/assets/a091c04d-ff99-4e08-8180-f6b2965f65a3)


- **ii)** Adjust Image Brightness.
  ![Screenshot 2025-05-20 200639](https://github.com/user-attachments/assets/b546d2d4-cf0b-4afe-ac22-826ba30d17e8)
  
  -**iii)** Modify
 ![Screenshot 2025-05-20 200646](https://github.com/user-attachments/assets/0776744a-8508-40f4-99f7-2c54dd4b7b07)

- **iv)** Generate Third Image Using Bitwise Operations.
![Screenshot 2025-05-20 200706](https://github.com/user-attachments/assets/2226ddc6-8dd3-499a-993d-41f04541c9b8)

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

