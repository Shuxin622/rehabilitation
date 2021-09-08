import cv2
import sys
import time

from kcf import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01


# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True


if __name__ == '__main__':

    cv2.namedWindow('tracking')
    tracker = kcftracker.KCFTracker()
    img = cv2.imread('../static/joint/images/video/0.jpg')
    x = 285
    y = 215
    w = 51
    h = 30
    tracker.init([x, y, w, h], img)
    index = 0
    while img is not None:
        peak_value, boundingbox = tracker.update(img)
        x = int(boundingbox[0])
        y = int(boundingbox[1])
        w = int(boundingbox[2])
        h = int(boundingbox[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('tracking', img)
        time.sleep(0.05)
        index += 1
        print('../static/joint/images/video/%d.jpg' % index)
        img = cv2.imread('../static/joint/images/video/%d.jpg' % index)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cv2.destroyAllWindows()

    # if (len(sys.argv) == 1):
    #     cap = cv2.VideoCapture(0)
    # elif (len(sys.argv) == 2):
    #     if (sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
    #         cap = cv2.VideoCapture(int(sys.argv[1]))
    #     else:
    #         cap = cv2.VideoCapture(sys.argv[1])
    #         inteval = 30
    # else:
    #     assert (0), "too many arguments"
    #
    # tracker = kcftracker.KCFTracker()  # hog, fixed_window, multiscale
    # # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    #
    # cv2.namedWindow('tracking')
    # cv2.setMouseCallback('tracking', draw_boundingbox)
    #
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     if (selectingObject):
    #         cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
    #     elif (initTracking):
    #         cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
    #
    #         tracker.init([ix, iy, w, h], frame)
    #
    #         initTracking = False
    #         onTracking = True
    #     elif (onTracking):
    #         peak_value, boundingbox = tracker.update(frame)
    #
    #         print(boundingbox)
    #         x = int(boundingbox[0])
    #         y = int(boundingbox[1])
    #         w = int(boundingbox[2])
    #         h = int(boundingbox[3])
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         # boundingbox = map(int, boundingbox)
    #         # cv2.rectangle(frame, (int(boundingbox[0]), int(boundingbox[1])),
    #         #               (int(boundingbox[0]) + int(boundingbox[2]), int(boundingbox[1]) + int(boundingbox[3])), (0, 255, 255), 1)
    #
    #     cv2.imshow('tracking', frame)
    #     c = cv2.waitKey(inteval) & 0xFF
    #     if c == 27 or c == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()