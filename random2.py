import numpy as np
import cv2
from matplotlib import pyplot as plt


PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.2,
    minDistance=15,
    blockSize=9
)

image_filter = PREVIEW
alive = True
# cv2.namedWindow("Camera Filters",cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(0)


while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        med = np.median(frame)

        result = cv2.Canny(frame, max(0,0.7*med), min(255,1.3*med))
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.int0(corners).reshape(-1, 2):
                cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

    cv2.imshow("Camera Filters", result)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY

    elif key == ord('b') or key == ord('B'):
        image_filter = BLUR

    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES

    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

source.release()
cv2.destroyAllWindows()
