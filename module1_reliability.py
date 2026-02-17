"""
Enhanced Module 1: Environmental Reliability Assessment
FIXED: Clean UI, no artifacts, stable display, professional overlay
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import deque


@dataclass
class EnvironmentContext:
    """Environmental assessment results in user-friendly format"""
    brightness_score: float  # 0-1 (0=too dark/bright, 1=perfect)
    contrast_score: float    # 0-1 (0=flat, 1=good contrast)
    edge_continuity: float   # 0-1 (0=no edges, 1=clear lanes)
    sharpness_score: float   # 0-1 (0=blurry, 1=sharp)
    weather_condition: str   # "Clear", "Rainy/Foggy", "Low Visibility"
    illumination_type: str   # "Daytime", "Night", "Overexposed"
    occlusion_level: float   # 0-1 (0=no obstruction, 1=blocked)
    reliability_score: float # 0-1 overall score
    confidence_flag: str     # "EXCELLENT", "GOOD", "FAIR", "POOR"


class LaneReliabilityAssessor:
    """
    Assesses environmental conditions for lane detection reliability
    IMPROVEMENTS:
    - Temporal smoothing for stable readings
    - Clean UI with proper alignment
    - No artifacts or debug text
    """
    def __init__(self):
        # Thresholds for assessment
        self.brightness_range = (50, 200)
        self.contrast_min = 40
        self.edge_threshold = 0.3
        self.blur_threshold = 100
        
        # Temporal smoothing buffers
        self.brightness_buffer = deque(maxlen=10)
        self.contrast_buffer = deque(maxlen=10)
        self.edge_buffer = deque(maxlen=10)
        self.sharpness_buffer = deque(maxlen=10)
        
    def smooth_metric(self, value, buffer):
        """Apply temporal smoothing to prevent UI flicker"""
        buffer.append(value)
        if len(buffer) >= 3:
            return float(np.median(list(buffer)))
        return value
        
    def assess_brightness(self, frame: np.ndarray) -> Tuple[float, str]:
        """Check if lighting is good for lane detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            illumination = "Night"
            score = mean_brightness / 50
        elif mean_brightness > 200:
            illumination = "Overexposed"
            score = (255 - mean_brightness) / 55
        else:
            illumination = "Daytime"
            score = 1.0
            
        score = self.smooth_metric(score, self.brightness_buffer)
        return min(max(score, 0.0), 1.0), illumination
    
    def assess_contrast(self, frame: np.ndarray) -> float:
        """Check if there's enough contrast to see lane markings"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        score = min(contrast / 60, 1.0)
        return self.smooth_metric(score, self.contrast_buffer)
    
    def assess_edge_continuity(self, frame: np.ndarray) -> float:
        """Check if lane markings have clear, continuous edges"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        h, w = edges.shape
        roi = edges[int(h*0.5):, :]
        
        edge_density = np.sum(roi > 0) / roi.size
        horizontal_proj = np.sum(roi, axis=1)
        continuity = np.count_nonzero(horizontal_proj > w * 0.1) / len(horizontal_proj)
        
        score = (edge_density * 0.5 + continuity * 0.5)
        score = self.smooth_metric(score, self.edge_buffer)
        return min(score * 3, 1.0)
    
    def assess_sharpness(self, frame: np.ndarray) -> float:
        """Check if image is sharp (not blurry)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(laplacian_var / 500, 1.0)
        return self.smooth_metric(score, self.sharpness_buffer)
    
    def detect_weather_conditions(self, frame: np.ndarray) -> Tuple[str, float]:
        """Detect weather conditions that might affect visibility"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        mid_concentration = np.sum(hist[80:180])
        contrast = gray.std()
        
        if mid_concentration > 0.6 and contrast < 35:
            return "Rainy/Foggy", mid_concentration
        elif contrast < 25:
            return "Low Visibility", 0.7
        else:
            return "Clear", 0.1
    
    def detect_occlusion(self, frame: np.ndarray) -> float:
        """Detect if view is blocked"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        h, w = v_channel.shape
        roi = v_channel[int(h*0.4):, :]
        
        dark_mask = roi < 40
        occlusion_ratio = np.sum(dark_mask) / roi.size
        
        return min(occlusion_ratio * 2, 1.0)
    
    def calculate_reliability_score(self, context: Dict) -> Tuple[float, str]:
        """Calculate overall reliability score"""
        weights = {
            'brightness': 0.20,
            'contrast': 0.20,
            'edge_continuity': 0.25,
            'sharpness': 0.15,
            'weather': 0.10,
            'occlusion': 0.10
        }
        
        score = (
            context['brightness_score'] * weights['brightness'] +
            context['contrast_score'] * weights['contrast'] +
            context['edge_continuity'] * weights['edge_continuity'] +
            context['sharpness_score'] * weights['sharpness'] +
            (1 - context['weather_impact']) * weights['weather'] +
            (1 - context['occlusion_level']) * weights['occlusion']
        )
        
        if score >= 0.80:
            confidence = "EXCELLENT"
        elif score >= 0.60:
            confidence = "GOOD"
        elif score >= 0.40:
            confidence = "FAIR"
        else:
            confidence = "POOR"
            
        return score, confidence
    
    def assess_frame(self, frame: np.ndarray) -> EnvironmentContext:
        """Main assessment function"""
        brightness_score, illumination = self.assess_brightness(frame)
        contrast_score = self.assess_contrast(frame)
        edge_continuity = self.assess_edge_continuity(frame)
        sharpness_score = self.assess_sharpness(frame)
        weather_condition, weather_impact = self.detect_weather_conditions(frame)
        occlusion_level = self.detect_occlusion(frame)
        
        context_dict = {
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'edge_continuity': edge_continuity,
            'sharpness_score': sharpness_score,
            'weather_impact': weather_impact,
            'occlusion_level': occlusion_level
        }
        
        reliability_score, confidence_flag = self.calculate_reliability_score(context_dict)
        
        return EnvironmentContext(
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            edge_continuity=edge_continuity,
            sharpness_score=sharpness_score,
            weather_condition=weather_condition,
            illumination_type=illumination,
            occlusion_level=occlusion_level,
            reliability_score=reliability_score,
            confidence_flag=confidence_flag
        )
    
    def draw_text_with_shadow(self, img, text, pos, font, scale, color, thickness):
        """Draw text with shadow for better readability"""
        x, y = pos
        cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    
    def visualize_assessment(self, frame: np.ndarray, context: EnvironmentContext) -> np.ndarray:
        """
        FIXED: Clean, professional visualization with:
        - Stable positioning
        - No artifacts or debug symbols
        - Proper alignment
        - Smooth transitions
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Color scheme based on reliability
        if context.reliability_score >= 0.80:
            bg_color = (0, 100, 50)
            text_color = (0, 255, 100)
            status_icon = "OK"
        elif context.reliability_score >= 0.60:
            bg_color = (0, 120, 100)
            text_color = (0, 255, 200)
            status_icon = "OK"
        elif context.reliability_score >= 0.40:
            bg_color = (0, 100, 150)
            text_color = (0, 200, 255)
            status_icon = "!"
        else:
            bg_color = (0, 30, 150)
            text_color = (0, 100, 255)
            status_icon = "!"
        
        # === TOP STATUS BAR ===
        overlay = vis_frame.copy()
        bar_height = 90
        
        cv2.rectangle(overlay, (0, 0), (w, bar_height), bg_color, -1)
        vis_frame = cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0)
        
        # Main status text (centered, clean)
        font = cv2.FONT_HERSHEY_DUPLEX
        status_text = f"DETECTION RELIABILITY: {context.confidence_flag}"
        
        text_size = cv2.getTextSize(status_text, font, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        self.draw_text_with_shadow(vis_frame, status_text, (text_x, 40), 
                                   font, 1.0, (255, 255, 255), 2)
        
        # Reliability percentage
        score_text = f"{context.reliability_score:.0%}"
        score_size = cv2.getTextSize(score_text, font, 1.2, 3)[0]
        score_x = (w - score_size[0]) // 2
        self.draw_text_with_shadow(vis_frame, score_text, (score_x, 75), 
                                   font, 1.2, text_color, 3)
        
        # === METRICS PANEL (RIGHT) ===
        panel_overlay = vis_frame.copy()
        panel_w = 380
        panel_h = 320
        panel_x = w - panel_w - 20
        panel_y = bar_height + 20
        
        cv2.rectangle(panel_overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (25, 25, 25), -1)
        vis_frame = cv2.addWeighted(panel_overlay, 0.85, vis_frame, 0.15, 0)
        cv2.rectangle(vis_frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (100, 100, 100), 2)
        
        # Panel title
        cv2.putText(vis_frame, "ENVIRONMENTAL METRICS", 
                   (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        
        # Metrics with bars (clean, aligned)
        metrics = [
            ("Brightness", context.brightness_score),
            ("Contrast", context.contrast_score),
            ("Edge Clarity", context.edge_continuity),
            ("Sharpness", context.sharpness_score),
        ]
        
        y_pos = panel_y + 70
        for name, score in metrics:
            # Metric name
            cv2.putText(vis_frame, name, (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Score bar
            bar_len = int(320 * score)
            bar_color = (0, 255, 100) if score >= 0.7 else (0, 200, 255) if score >= 0.4 else (0, 100, 255)
            
            cv2.rectangle(vis_frame, (panel_x + 20, y_pos + 10), 
                         (panel_x + 340, y_pos + 24), (50, 50, 50), -1)
            cv2.rectangle(vis_frame, (panel_x + 20, y_pos + 10), 
                         (panel_x + 20 + bar_len, y_pos + 24), bar_color, -1)
            
            # Score percentage
            cv2.putText(vis_frame, f"{score:.0%}", (panel_x + 345, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1, cv2.LINE_AA)
            
            y_pos += 65
        
        # === INFO BOX (LEFT) ===
        rec_overlay = vis_frame.copy()
        rec_w = 420
        rec_h = 120
        rec_x = 20
        rec_y = bar_height + 20
        
        cv2.rectangle(rec_overlay, (rec_x, rec_y), 
                     (rec_x + rec_w, rec_y + rec_h), 
                     (25, 25, 25), -1)
        vis_frame = cv2.addWeighted(rec_overlay, 0.85, vis_frame, 0.15, 0)
        cv2.rectangle(vis_frame, (rec_x, rec_y), 
                     (rec_x + rec_w, rec_y + rec_h), 
                     (100, 100, 100), 2)
        
        # Recommendation
        if context.reliability_score >= 0.80:
            recommendation = "EXCELLENT conditions for detection"
            advice = "All systems optimal"
        elif context.reliability_score >= 0.60:
            recommendation = "GOOD conditions for detection"
            advice = "Minor issues - detection reliable"
        elif context.reliability_score >= 0.40:
            recommendation = "FAIR conditions - reduced reliability"
            advice = "Moderate issues - use caution"
        else:
            recommendation = "POOR conditions - not recommended"
            advice = "Significant issues detected"
        
        cv2.putText(vis_frame, "SYSTEM RECOMMENDATION:", 
                   (rec_x + 15, rec_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        
        cv2.putText(vis_frame, recommendation, 
                   (rec_x + 15, rec_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
        
        cv2.putText(vis_frame, advice, 
                   (rec_x + 15, rec_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        
        # === BOTTOM INFO BAR ===
        info_overlay = vis_frame.copy()
        info_height = 50
        cv2.rectangle(info_overlay, (0, h - info_height), (w, h), (20, 20, 20), -1)
        vis_frame = cv2.addWeighted(info_overlay, 0.75, vis_frame, 0.25, 0)
        
        # Clean status text (left-aligned, no artifacts)
        status_line = f"Weather: {context.weather_condition} | Lighting: {context.illumination_type}"
        cv2.putText(vis_frame, status_line, (25, h - 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        
        return vis_frame


def process_image(image_path: str, output_path: str):
    """Process a single image"""
    assessor = LaneReliabilityAssessor()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Analyzing environmental conditions...")
    context = assessor.assess_frame(frame)
    vis_frame = assessor.visualize_assessment(frame, context)
    
    cv2.imwrite(output_path, vis_frame)
    
    print(f"Image processed: {output_path}")
    print(f"Reliability: {context.reliability_score:.0%} ({context.confidence_flag})")


def process_video(video_path: str, output_path: str):
    """Process a video file"""
    assessor = LaneReliabilityAssessor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames} | FPS: {fps}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        context = assessor.assess_frame(frame)
        vis_frame = assessor.visualize_assessment(frame, context)
        out.write(vis_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames} | Reliability: {context.reliability_score:.0%}")
    
    cap.release()
    out.release()
    print(f"Video processed: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python module1_reliability.py <input> <output>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            process_video(input_path, output_path)
        else:
            process_image(input_path, output_path)