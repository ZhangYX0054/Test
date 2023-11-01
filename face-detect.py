import imutils
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import argparse
import cv2
import dlib
import time


#  基本参数
EYE_AR_THRESH = 0.3
EYE_AR_CONSE_FRAMES = 48
COUNTER = 0
ALARM_ON = False


ap = argparse.ArgumentParser
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to the shape-predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path  alarm.WAV file")
args = vars(ap.parse_args)


def sound_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


print("[INFO]loading facial landmarks......")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(argparse['shape-predictor'])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left-eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right-eye']

print("[INFO]start video......")

vs = VideoStream(src=args['webcam']).start()
time.sleep(1.0)  # 延迟预热
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart, lEnd]
        rightEye = shape[rStart, rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0  # 平均纵横比计算

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSE_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True

                    if args['alarm'] != '':
                        t = Thread(target=sound_alarm, args=(args[''], ))
                        t.daemon = True

                cv2.putText(frame, 'ALARM！！！', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, 'ERA:{:.2f}'.format(ear),  (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyWindow()
vs.stop()

