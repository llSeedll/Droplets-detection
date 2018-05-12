from time import sleep
import cv2
import numpy as np

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

VideoPath = 'example.mp4'
#Path to video, grabbed by CV
VideoGrab = cv2.VideoCapture(VideoPath)

#Debugger, waiting for vid
while not VideoGrab.isOpened():
    VideoGrab = cv2.VideoCapture(VideoPath)
    cv2.waitKey(1000) #waitKey waits for a pressed key; the delay is 1s.
    print("Wait for the file to be available :)") #To indicate path is incorrect, or something wrong with video.

#Define the Frames of the VideoGrab
PosFrame = VideoGrab.get(cv2.CAP_PROP_POS_FRAMES)

firstFrame = None
#===========================================================================================
# DETECTION
#===========================================================================================
count = 0
frameCount = 0
prevFirstX = 500
while True:
    flag, Frame = VideoGrab.read() #Reading the Frames

    if flag:
        x1 = 200
        y1 = 140
        x2 = 400
        y2 = 215
        gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (51, 51), 0)
        gray = cv2.Canny(Frame,255,255)
        cv2.rectangle(Frame, (x1, y1), (x2, y2), (0,255,255), 2)

        # compute the absolute difference between the current frame and
        # first frame
        if firstFrame is None:
            firstFrame = gray
            continue
        #frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
     
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        #thresh = cv2.dilate(thresh, None, iterations=2)
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        found = False
        firstX = 500
        for c in cnts:
            # if the contour is too small, ignore it
            area = cv2.contourArea(c)
            if area < 35:
                continue
     
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if w > 50:
                continue
            if ((x >= x1 and y>=y1 and (x + w) <= x2 and (y + h) <= y2) or 
                (((x + w) >= x1 and (x < x1) and y>=y1 and (y + h) <= y2) or ((x + w) > x2) and (x <= x2)) and y>=y1 and (y + h) <= y2 ):
                if (x < firstX) :
                    firstX = x
                    found = True
                cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if found and firstX < prevFirstX:
            count += 1
            print("Element detected: {}".format(count))

        if found:
            prevFirstX = firstX
        frameCount += 1

        #Frame = Frame[y1:y2, x1:x2]
        cv2.imshow('Video', Frame) #Shows the video
        PosFrame = VideoGrab.get(cv2.CAP_PROP_POS_FRAMES)
    else:
        VideoGrab.set(cv2.CAP_PROP_POS_FRAMES, PosFrame-1)
        print("Frame is not ready")
        cv2.waitKey(1000) #Otherwise it does not run the video and counts back one.

#Break options: waitKey intrinsically requires an escape option by the user (hence decimal 27 = escape)
    if cv2.waitKey(10) == 27:
        break

#Here we break when the number of frames = number of frames in the video, i.e. stops when the video does
    if VideoGrab.get(cv2.CAP_PROP_POS_FRAMES) == VideoGrab.get(cv2.CAP_PROP_FRAME_COUNT):
        break

    firstFrame = gray
    #sleep(.1)
#File I/O
VideoGrab.release()
cv2.destroyAllWindows()