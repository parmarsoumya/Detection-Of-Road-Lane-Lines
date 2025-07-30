import cv2
import numpy as np 

def detect_lane(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height = image.shape[0]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (100, height), (image.shape[1] - 100, height), (image.shape[1] // 2, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
    line_img = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    combined = cv2.addWeighted(image, 0.8, line_img, 1, 1)
    return combined

USE_IMAGE = True

if USE_IMAGE:
    img = cv2.imread(r'C:\Users\shalu\OneDrive\Documents\PinnacleLabs\Task 2\test_image.jpg')

    if img is None:
        print("Image not found. Make sure 'test_image.jpg' is in the folder.")
    else:
        result = detect_lane(img)
        cv2.imshow("Detected Lane Lines", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
