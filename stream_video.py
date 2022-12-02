import time
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)
# cap = cv2.VideoCapture('Video1.mp4')
cap = cv2.VideoCapture('marco.avi')

def gen():

    while True:
        ret, frame = cap.read()

        frame_counter = 0
        frame_counter += 1

        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
               frame_counter = 0  # Or whatever as long as it is the same as next line
               cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
          # ####################

        if not ret:
            break

        else:
            # suc, encode = cv2.imencode('.jpg', frame)
            # frame = encode.tobytes()
            framegris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask_marco = cv2.threshold(framegris, 80, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((6, 6), np.uint8)
            mask_marco = cv2.erode(mask_marco, kernel, 20)
            mask_marco = cv2.dilate(mask_marco, kernel, 1)

            height, width = frame.shape[:2]

            cnts = cv2.findContours(mask_marco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            # print(cnts)
            for c in cnts:
                epsilon = 0.01 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                # print("approx", approx)
                if len(approx) == 4:

                    aspect_ratio = float(width) / height
                    if aspect_ratio == 1:
                        print('Cuadrado')
                    else:

                        cv2.drawContours(frame, [approx], 0, (0, 0, 0), 2)

                        cv2.circle(frame, tuple(approx[0][0]), 7, (0, 255, 0), 2)
                        cv2.circle(frame, tuple(approx[3][0]), 7, (255, 255, 0), 2)
                        cv2.circle(frame, tuple(approx[1][0]), 7, (255, 0, 0), 2)
                        cv2.circle(frame, tuple(approx[2][0]), 7, (0, 0, 255), 2)
                        pts1 = np.float32([tuple(approx[0][0]), tuple(approx[3][0]), tuple(approx[1][0]), tuple(approx[2][0])])
                        pts2 = np.float32([[0, 0], [int(width), 0], [0, int(height)], [int(width), int(height)]])
                        M = cv2.getPerspectiveTransform(pts1, pts2)
                        dst = cv2.warpPerspective(frame, M, (int(width), int(height)))
                        # cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
                        # cv2.imshow("dst", dst)
                        suc, encode = cv2.imencode('.jpg', dst)
                        dst = encode.tobytes()

                        # if approx[0][0][0] != 0 and approx[0][0][1] != 0 and approx[1][0][0] != 0 and approx[1][0][1] != 0 and \
                        #     approx[2][0][0] != 0 and approx[2][0][1] != 0 and approx[3][0][0] != 0 and approx[3][0][1] != 0:
                        #     print("####################")
                        #     print(approx[0][0][0])
                        #     print(approx[0][0][1])
                        #     print(approx[1][0][0])
                        #     print(approx[1][0][1])
                        #     print(approx)
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + dst + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
     app.run(host='0.0.0.0',port=5000)
