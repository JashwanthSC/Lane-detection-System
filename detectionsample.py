import cv2
import numpy as np
from collections import deque

# ============================================
# CONFIG
# ============================================
VIDEO_PATH = "test_video.mp4"
SMOOTHING_FRAMES = 7
# ============================================


class LaneDetector:
    def __init__(self):
        self.left_fit_hist = deque(maxlen=SMOOTHING_FRAMES)
        self.right_fit_hist = deque(maxlen=SMOOTHING_FRAMES)

    # ---------- Thresholding ----------
    def threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s = hls[:, :, 2]

        sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255 * sobelx / np.max(sobelx))

        _, sobel_bin = cv2.threshold(sobelx, 40, 255, cv2.THRESH_BINARY)
        _, s_bin = cv2.threshold(s, 120, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(sobel_bin, s_bin)
        return combined

    # ---------- Perspective Transform ----------
    def warp(self, img):
        h, w = img.shape[:2]

        src = np.float32([
            [w * 0.43, h * 0.65],
            [w * 0.58, h * 0.65],
            [w * 0.10, h * 0.95],
            [w * 0.95, h * 0.95]
        ])

        dst = np.float32([
            [w * 0.25, 0],
            [w * 0.75, 0],
            [w * 0.25, h],
            [w * 0.75, h]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, M, (w, h))
        return warped, Minv

    # ---------- Sliding Window ----------
    def detect_lane(self, binary):
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = binary.shape[0] // nwindows
        margin = 100
        minpix = 50

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_inds = []
        right_inds = []

        for window in range(nwindows):
            win_y_low = binary.shape[0] - (window + 1) * window_height
            win_y_high = binary.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_inds.append(good_left)
            right_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)

        left_fit = np.polyfit(nonzeroy[left_inds], nonzerox[left_inds], 2)
        right_fit = np.polyfit(nonzeroy[right_inds], nonzerox[right_inds], 2)

        self.left_fit_hist.append(left_fit)
        self.right_fit_hist.append(right_fit)

        return np.mean(self.left_fit_hist, axis=0), np.mean(self.right_fit_hist, axis=0)

    # ---------- Draw Lane ----------
    def draw_lane(self, img, binary, left_fit, right_fit, Minv):
        h, w = binary.shape
        ploty = np.linspace(0, h - 1, h)

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        overlay = np.zeros_like(img)
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])

        cv2.fillPoly(overlay, np.int32([np.hstack((pts_left, pts_right))]), (0, 255, 0))
        overlay = cv2.warpPerspective(overlay, Minv, (w, h))

        return cv2.addWeighted(img, 1, overlay, 0.3, 0)


# ============================================
# MAIN
# ============================================
cap = cv2.VideoCapture(VIDEO_PATH)
lane_detector = LaneDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    binary = lane_detector.threshold(frame)
    warped, Minv = lane_detector.warp(binary)

    try:
        left_fit, right_fit = lane_detector.detect_lane(warped)
        result = lane_detector.draw_lane(frame, warped, left_fit, right_fit, Minv)
    except:
        result = frame

    cv2.imshow("Lane Detection Test", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
