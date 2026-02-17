"""
Module 2: Lane Detection
Clear tracking line anchored to road perspective
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class Lane:
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    left_curve: float
    right_curve: float
    center_offset: float
    lane_width: float
    confidence: float


class CameraCalibrator:
    def undistort(self, img):
        return img


class LaneDetector:
    def __init__(self, calibrate=False):
        self.img_width = 1280
        self.img_height = 720
        self.calibrator = CameraCalibrator()

        # Perspective transform
        self.src = np.float32([
            [580, 460],
            [700, 460],
            [1100, 720],
            [200, 720]
        ])

        self.dst = np.float32([
            [300, 0],
            [980, 0],
            [980, 720],
            [300, 720]
        ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

        self.left_hist = deque(maxlen=10)
        self.right_hist = deque(maxlen=10)

        self.last_good_left = None
        self.last_good_right = None

    def combined_threshold(self, img):
        """Clean thresholding"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobelx = np.uint8(255 * np.abs(sobelx) / (np.max(np.abs(sobelx)) + 1e-6))
        grad = ((sobelx >= 20) & (sobelx <= 100)).astype(np.uint8)

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_bin = (hls[:, :, 2] >= 170).astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        b_bin = ((lab[:, :, 2] >= 155) & (lab[:, :, 2] <= 200)).astype(np.uint8)

        combined = np.zeros_like(grad)
        combined[(grad == 1) | (s_bin == 1) | (b_bin == 1)] = 1
        return combined

    def detect_lanes(self, frame):
        """Main detection pipeline"""
        img = cv2.resize(frame, (self.img_width, self.img_height))
        binary = self.combined_threshold(img)
        warped = cv2.warpPerspective(binary, self.M, (self.img_width, self.img_height))

        histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = warped.shape[0] // self.nwindows
        nonzeroy, nonzerox = warped.nonzero()

        leftx_current, rightx_current = leftx_base, rightx_base
        left_inds, right_inds = [], []

        for window in range(self.nwindows):
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height

            win_xl_low = leftx_current - self.margin
            win_xl_high = leftx_current + self.margin
            win_xr_low = rightx_current - self.margin
            win_xr_high = rightx_current + self.margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xl_low) & (nonzerox < win_xl_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xr_low) & (nonzerox < win_xr_high)).nonzero()[0]

            left_inds.append(good_left)
            right_inds.append(good_right)

            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)

        if len(left_inds) < 200 or len(right_inds) < 200:
            if self.last_good_left is not None:
                return Lane(self.last_good_left, self.last_good_right, 0, 0, 0, 3.7, 0.4)
            return Lane(None, None, 0, 0, 0, 0, 0)

        left_fit = np.polyfit(nonzeroy[left_inds], nonzerox[left_inds], 2)
        right_fit = np.polyfit(nonzeroy[right_inds], nonzerox[right_inds], 2)

        self.left_hist.append(left_fit)
        self.right_hist.append(right_fit)

        left_fit = np.mean(self.left_hist, axis=0)
        right_fit = np.mean(self.right_hist, axis=0)

        self.last_good_left = left_fit
        self.last_good_right = right_fit

        xm_per_pix = 3.7 / 700
        y_eval = self.img_height

        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

        center_offset = ((self.img_width / 2) - (left_x + right_x) / 2) * xm_per_pix
        lane_width = (right_x - left_x) * xm_per_pix

        return Lane(left_fit, right_fit, 0, 0, center_offset, lane_width, 1.0)

    def visualize_lanes(self, frame, lane):
        """
        Visualization with clear tracking line anchored to road
        """
        img = cv2.resize(frame, (self.img_width, self.img_height))
        h, w = img.shape[:2]

        if lane.left_fit is None:
            result = img.copy()
            cv2.rectangle(result, (20, h-80), (350, h-20), (0, 0, 0), -1)
            cv2.rectangle(result, (20, h-80), (350, h-20), (0, 0, 255), 2)
            cv2.putText(result, "NO LANES DETECTED", (35, h-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return result

        # === DRAW IN BIRD'S EYE VIEW ===
        ploty = np.linspace(0, h - 1, h)
        
        leftx = lane.left_fit[0]*ploty**2 + lane.left_fit[1]*ploty + lane.left_fit[2]
        rightx = lane.right_fit[0]*ploty**2 + lane.right_fit[1]*ploty + lane.right_fit[2]

        # Create overlay in bird's eye view
        lane_overlay = np.zeros((h, w, 3), dtype=np.uint8)

        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Light green fill
        cv2.fillPoly(lane_overlay, np.int32([pts]), (0, 100, 0))
        
        # Lane boundaries (thinner)
        cv2.polylines(lane_overlay, np.int32([pts_left]), False, (255, 100, 0), 8)
        cv2.polylines(lane_overlay, np.int32([pts_right]), False, (0, 100, 255), 8)

        # === TRACKING LINE IN BIRD'S EYE VIEW (BEFORE WARPING) ===
        # This ensures it stays anchored to the road
        tracking_y_bev = int(h * 0.55)  # 55% down in bird's eye = proper road position
        
        # Calculate center at tracking position
        left_x_track = lane.left_fit[0]*tracking_y_bev**2 + lane.left_fit[1]*tracking_y_bev + lane.left_fit[2]
        right_x_track = lane.right_fit[0]*tracking_y_bev**2 + lane.right_fit[1]*tracking_y_bev + lane.right_fit[2]
        center_x = int((left_x_track + right_x_track) / 2)
        
        # Calculate line endpoints (stay within lane)
        line_width_ratio = 0.30  # 30% of lane width
        half_line = int((right_x_track - left_x_track) * line_width_ratio)
        
        # Draw thick tracking line in bird's eye view
        # Yellow/green line with white border
        cv2.line(lane_overlay,
                (center_x - half_line, tracking_y_bev),
                (center_x + half_line, tracking_y_bev),
                (255, 255, 255), 20)  # White border (thick)
        
        cv2.line(lane_overlay,
                (center_x - half_line, tracking_y_bev),
                (center_x + half_line, tracking_y_bev),
                (0, 255, 0), 14)  # Green line

        # === WARP BACK TO CAMERA VIEW ===
        # Now the line will be perspective-correct and anchored to road
        unwarped = cv2.warpPerspective(lane_overlay, self.Minv, (w, h))

        # Blend with original image
        result = cv2.addWeighted(img, 1.0, unwarped, 0.35, 0)

        # === SIMPLE INFO DISPLAY ===
        
        offset_cm = abs(lane.center_offset * 100)
        
        if lane.center_offset < -0.05:
            position = "LEFT"
            pos_color = (255, 200, 0)
        elif lane.center_offset > 0.05:
            position = "RIGHT"
            pos_color = (0, 200, 255)
        else:
            position = "CENTER"
            pos_color = (0, 255, 0)

        # Top left - Status
        cv2.rectangle(result, (10, 10), (260, 55), (0, 0, 0), -1)
        cv2.putText(result, "LANE TRACKING", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Bottom left - Position
        cv2.rectangle(result, (10, h-110), (420, h-10), (0, 0, 0), -1)
        cv2.rectangle(result, (10, h-110), (420, h-10), (100, 100, 100), 2)
        
        cv2.putText(result, f"Position: {position}", (20, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, pos_color, 2)
        
        cv2.putText(result, f"Offset: {offset_cm:.0f} cm | Width: {lane.lane_width:.1f}m", 
                   (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return result


def process_video(video_path, output_path):
    detector = LaneDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
    frame_count = 0
    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lane = detector.detect_lanes(frame)
        out.write(detector.visualize_lanes(frame, lane))
        frame_count += 1
        if frame_count % 30 == 0:
            status = "Detected" if lane.left_fit is not None else "Lost"
            print(f"Frame {frame_count} | Status: {status}")
    cap.release()
    out.release()
    print(f"Video saved: {output_path}")


def process_image(image_path, output_path):
    detector = LaneDetector()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return
    print(f"Processing image: {image_path}")
    lane = detector.detect_lanes(frame)
    result = detector.visualize_lanes(frame, lane)
    cv2.imwrite(output_path, result)
    if lane.left_fit is not None:
        print(f"Image saved: {output_path}")
        print(f"Offset: {abs(lane.center_offset * 100):.0f} cm | Width: {lane.lane_width:.2f} m")
    else:
        print(f"Image saved: {output_path} (no lanes detected)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python module2_detection.py <input> <output>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            process_video(input_path, output_path)
        else:
            process_image(input_path, output_path)