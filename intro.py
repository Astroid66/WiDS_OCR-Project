import cv2
import pytesseract

img = cv2.imread('image.jpg')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\AppData\Local\Programs\tesseract\tesseract.exe'
img2 = pytesseract.image_to_string(img, config=custom_config)

print(img2)