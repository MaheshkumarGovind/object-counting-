import cv2
import numpy as np
vid = cv2.VideoCapture(0)
detector = cv2.SimpleBlobDetector_create()
while (True):
    ret, image = vid.read()

    keypoints = detector.detect(image)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)

    text = "total number of pipes:" + str(len(keypoints))

    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow('blobs using default parameters',blobs)
    # cv2.waitKey()

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10;

    params.maxThreshold = 200;
    params.filterByArea = True
    params.minArea = 400

    params.filterByCircularity = True

    params.minCircularity = 0.1
    params.filterByConvexity = True 
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    blank = np.zeros((1, 1))
    image = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)

    text =str(len(keypoints))
    print(text)
    cv2.putText(image, text, (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
