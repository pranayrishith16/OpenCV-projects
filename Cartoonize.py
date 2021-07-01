import numpy as np
import cv2

def cartoonize_img(img, ds_factor=4, sketch_mode=False):
    # converting to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # median blur
    img_gray = cv2.medianBlur(img_gray, 7)
    # Detech edges using laplacian algo
    edges = cv2.Laplacian(img_gray,cv2.CV_8UC1,ksize=5)
    # Threshold
    ret,mask = cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)
    if sketch_mode:
        return cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5
    # Apply bilateral filter multiple times
    for i in range(num_repetitions):
        img = cv2.bilateralFilter(img,size,sigma_color,sigma_space)
    dst = np.zeros(img_gray.shape)
    dst = cv2.bitwise_and(img,img,mask=mask)
    return dst


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

curr_char = -1
prev_char = -1

while True:
    _, frame = cap.read()
    c = cv2.waitKey(1)
    if c == 27:
        break
    if c > -1 and c != prev_char:
        curr_char = c
    prev_char = c

    if curr_char == ord('s'):
        cv2.imshow("Cartoonize", cartoonize_img(frame, sketch_mode=True))
    elif curr_char == ord('c'):
        cv2.imshow("Cartoonize", cartoonize_img(frame, sketch_mode=False))
    else:
        cv2.imshow("Cartoonize", frame)


cap.release()
cv2.destroyAllWindows()
