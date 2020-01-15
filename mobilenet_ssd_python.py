# Import the neccesary libraries
import imutils
import numpy as np
import argparse
import cv2
import datetime
from threading import Thread

password = 'Babl!@#456'

url_1 = 'rtsp://admin:' + password + '@192.168.10.2:554'
url_2 = 'rtsp://admin:' + password + '@192.168.10.3:554'
url_3 = 'rtsp://admin:' + password + '@192.168.10.4:554'
url_4 = 'rtsp://admin:' + password + '@192.168.10.6:554'

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                    help='Path to text network file: '
                         'MobileNetSSD_deploy.prototxt for Caffe model or '
                    )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: '
                         'MobileNetSSD_deploy.caffemodel for Caffe model or '
                    )
parser.add_argument("--thr", default=0.3, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
# classNames = {0: 'background',
#              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
#              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
#              10: 'cow', 12: 'dog', 13: 'horse',
#               14: 'motorbike', 15: 'person', 16: 'pottedplant',
#              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

classNames = {0: 'background',
              15: 'person'}


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 5

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# Open video file or capture device.
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    capture_1 = WebcamVideoStream(url_1).start()
    capture_2 = WebcamVideoStream(url_2).start()
    capture_3 = WebcamVideoStream(url_3).start()
    capture_4 = WebcamVideoStream(url_4).start()

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:

    read_correct = True

    frame_1 = capture_1.read()
    frame_2 = capture_2.read()
    frame_3 = capture_3.read()
    frame_4 = capture_4.read()

    if (frame_1 is not None) and (frame_2 is not None) and (frame_3 is not None) and (frame_4 is not None):
        frame_1 = cv2.resize(frame_1, (480, 270))
        frame_2 = cv2.resize(frame_2, (480, 270))
        frame_3 = cv2.resize(frame_3, (480, 270))
        frame_4 = cv2.resize(frame_4, (480, 270))

        top = np.hstack((frame_1, frame_2))
        bot = np.hstack((frame_3, frame_4))
        output = np.vstack((top, bot))

        # Capture frame-by-frame
        # ret, frame = output.read()
        frame = output
        frame_resized = cv2.resize(frame, (500, 500))  # resize frame for prediction
        # frame_resized = output

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has different size.
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (500, 500), (127.5, 127.5, 127.5), False)
        # Set to network the input blob
        net.setInput(blob)
        # Prediction of network
        detections = net.forward()

        # Size of frame resize (300x300)
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]

        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > args.thr:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label

                # Object location
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame
                heightFactor = frame.shape[0] / 500.0
                widthFactor = frame.shape[1] / 500.0
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                # Draw location of object
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 0, 255))

                # Draw label and confidence of prediction in frame resized
                """
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
    
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    #cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  #(xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  #(255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
                    print(label)  # print class and confidence
                """

    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(27) >= 0:  # Break with ESC
        break
