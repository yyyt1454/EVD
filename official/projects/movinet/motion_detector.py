
import cv2 

"""
* Process * 
1) Input (clip): [1, 5, 224, 224, 3]
2) Take 1,3,5th frame and preprocess (convert to grayscale, apply gaussian blur) 
3) Get difference (1-3) and (3-5)
4) Apply bitwiseand operation to find intersection 
5) Check whether they are over threshold 
6) Find contours and pass small contours 
7) Return motion detection result 

"""

def preprocessing(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return frame 

def motion_detector(clip, threshold = 1):

    a,b,c = clip[0][0], clip[0][2], clip[0][4] 
    a,b,c = preprocessing(a), preprocessing(b), preprocessing(c)
    
    diff1 = cv2.absdiff(a, b)
    diff2 = cv2.absdiff(b, c)

    ret, diff1_t = cv2.threshold(diff1, threshold, 255, cv2.THRESH_BINARY)
    ret, diff2_t = cv2.threshold(diff2, threshold, 255, cv2.THRESH_BINARY)

    diff = cv2.bitwise_and(diff1_t, diff2_t)
    diff = diff.astype(np.uint8)

    contours, _ = cv2.findContours(image=diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)



    for contour in contours:
        if cv2.contourArea(contour) < 50:
            # too small: skip!
            continue
        else:
            return True

    return False 