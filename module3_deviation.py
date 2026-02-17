"""
Enhanced Module 3: Safety Monitoring and Driver Alert System
FIXES:
- Stable alerts with temporal filtering
- No false triggers from camera shake
- Clean UI with proper alignment
- Intelligent alert thresholds with context awareness
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List


class SafetyLevel(Enum):
    """Clear safety status"""
    SAFE = "Safe"
    CAUTION = "Caution"
    WARNING = "Warning"
    DANGER = "Danger"


@dataclass
class VehicleState:
    """Vehicle information"""
    speed_kmh: float
    steering_angle_deg: float


@dataclass
class SafetyAnalysis:
    """Analysis results with user-friendly data"""
    offset_cm: float
    position_text: str
    drift_rate_cm_s: float
    drift_direction: str
    safety_level: SafetyLevel
    alert_message: str
    sustained_drift: bool
    camera_stable: bool
    detection_quality_percent: float
    speed_kmh: float
    steering_angle_deg: float
    lane_width_m: float
    offset_history: List[float]


class CameraStabilityFilter:
    """
    IMPROVED: Robust filtering to eliminate false alerts
    - Multi-stage filtering
    - Shake detection
    - Smooth offset estimation
    """
    def __init__(self):
        self.offset_buffer = deque(maxlen=30)
        self.velocity_buffer = deque(maxlen=15)
        
        # Thresholds
        self.shake_threshold = 0.12  # meters
        self.velocity_threshold = 0.08  # m/s
        self.stable_frames_needed = 15  # 0.5 seconds at 30fps
        self.stable_counter = 0

    def is_camera_stable(self, offset):
        """Check if camera is stable (not shaking)"""
        self.offset_buffer.append(offset)
        
        if len(self.offset_buffer) < 10:
            return False

        # === CHECK 1: Rapid oscillations (primary shake indicator) ===
        recent = list(self.offset_buffer)[-10:]
        changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        
        if np.mean(changes) > self.shake_threshold:
            self.stable_counter = 0
            return False

        # === CHECK 2: Zigzag pattern (secondary shake indicator) ===
        if len(self.offset_buffer) >= 7:
            last = list(self.offset_buffer)[-7:]
            direction_changes = 0
            for i in range(1, len(last) - 1):
                if (last[i] - last[i - 1]) * (last[i + 1] - last[i]) < 0:
                    direction_changes += 1
            if direction_changes >= 4:
                self.stable_counter = 0
                return False

        # === CHECK 3: Velocity consistency ===
        if len(self.offset_buffer) >= 5:
            velocities = []
            recent_5 = list(self.offset_buffer)[-5:]
            for i in range(1, len(recent_5)):
                vel = abs(recent_5[i] - recent_5[i-1])
                velocities.append(vel)
            
            if np.std(velocities) > self.velocity_threshold:
                self.stable_counter = 0
                return False

        # Camera is stable
        self.stable_counter += 1
        return self.stable_counter >= self.stable_frames_needed

    def get_smoothed_offset(self):
        """Get smoothed offset value with outlier rejection"""
        if len(self.offset_buffer) >= 7:
            # Use median of recent values (robust to outliers)
            recent = list(self.offset_buffer)[-7:]
            return float(np.median(recent))
        elif len(self.offset_buffer) >= 3:
            return float(np.median(list(self.offset_buffer)))
        return self.offset_buffer[-1] if self.offset_buffer else 0.0


class SafetyMonitor:
    """
    IMPROVED: Stable safety monitoring with:
    - Multi-stage temporal filtering
    - Context-aware thresholds
    - Intelligent alert logic
    """
    def __init__(self):
        # Safety thresholds (METERS from lane center)
        self.SAFE_ZONE = 0.30       # Within 30cm - excellent
        self.CAUTION_ZONE = 0.50    # 30-50cm - acceptable
        self.WARNING_ZONE = 0.80    # 50-80cm - concerning
        # Beyond 80cm is DANGER
        
        # System components
        self.stability_filter = CameraStabilityFilter()
        self.offset_history = deque(maxlen=90)  # 3 seconds at 30fps
        
        # Sustained deviation tracking
        self.sustained_frames_threshold = 50  # 1.67 seconds at 30fps
        self.deviation_counter = 0
        self.last_deviation_direction = None
        
        # Alert state management
        self.alert_cooldown = 0
        self.last_alert_level = SafetyLevel.SAFE

    def analyze_driving_safety(self, raw_offset, lane_width, vehicle_state, detection_confidence=1.0):
        """
        IMPROVED: Main analysis with stable alert generation
        """
        
        # === STEP 1: STABILIZE READINGS ===
        camera_stable = self.stability_filter.is_camera_stable(raw_offset)
        stabilized_offset = self.stability_filter.get_smoothed_offset()
        self.offset_history.append(stabilized_offset)
        
        # === STEP 2: CALCULATE DRIFT RATE (SMOOTHED) ===
        drift_rate_m_s = 0.0
        if len(self.offset_history) >= 20:  # 0.67 seconds
            recent = list(self.offset_history)[-20:]
            # Use linear regression for smoother drift rate
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            drift_rate_m_s = coeffs[0] * 30  # Convert to m/s (assuming 30fps)
        
        # === STEP 3: DETERMINE POSITION ===
        offset_cm = stabilized_offset * 100
        
        if abs(stabilized_offset) < 0.05:
            position_text = "Center"
        elif stabilized_offset < 0:
            position_text = "Left"
        else:
            position_text = "Right"
        
        # Drift direction
        if abs(drift_rate_m_s) < 0.01:
            drift_direction = "Stable"
        elif drift_rate_m_s < 0:
            drift_direction = "Drifting Left"
        else:
            drift_direction = "Drifting Right"
        
        # === STEP 4: CHECK FOR SUSTAINED DEVIATION ===
        sustained_drift = self._check_sustained_deviation(
            stabilized_offset, camera_stable, position_text
        )
        
        # === STEP 5: DETERMINE SAFETY LEVEL (WITH ALERT HYSTERESIS) ===
        safety_level, alert_message = self._assess_safety_level(
            stabilized_offset, sustained_drift, vehicle_state.steering_angle_deg,
            camera_stable
        )
        
        # === STEP 6: ALERT COOLDOWN LOGIC ===
        # Prevent alert flapping
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1
            # Keep previous alert if still relevant
            if self.last_alert_level.value in ["Warning", "Danger"]:
                safety_level = self.last_alert_level
        else:
            self.last_alert_level = safety_level
            if safety_level.value in ["Warning", "Danger"]:
                self.alert_cooldown = 15  # Cooldown for 0.5 seconds
        
        # === STEP 7: ADJUST CONFIDENCE ===
        stability_factor = 0.9 if camera_stable else 0.5
        quality_percent = stability_factor * detection_confidence * 100
        
        # === STEP 8: COMPILE RESULTS ===
        return SafetyAnalysis(
            offset_cm=offset_cm,
            position_text=position_text,
            drift_rate_cm_s=drift_rate_m_s * 100,
            drift_direction=drift_direction,
            safety_level=safety_level,
            alert_message=alert_message,
            sustained_drift=sustained_drift,
            camera_stable=camera_stable,
            detection_quality_percent=quality_percent,
            speed_kmh=vehicle_state.speed_kmh,
            steering_angle_deg=vehicle_state.steering_angle_deg,
            lane_width_m=lane_width,
            offset_history=list(self.offset_history)
        )

    def _check_sustained_deviation(self, offset, stable, position):
        """Check if vehicle has been drifting for extended period"""
        if not stable:
            self.deviation_counter = max(0, self.deviation_counter - 3)
            return False

        # Count frames with significant deviation
        if abs(offset) > self.CAUTION_ZONE:
            if self.last_deviation_direction == position:
                self.deviation_counter += 1
            else:
                self.deviation_counter = 1
                self.last_deviation_direction = position
        else:
            self.deviation_counter = max(0, self.deviation_counter - 2)

        return self.deviation_counter >= self.sustained_frames_threshold

    def _assess_safety_level(self, offset, sustained, steering_angle, camera_stable):
        """
        IMPROVED: Context-aware safety assessment with hysteresis
        """
        abs_offset = abs(offset)
        abs_steering = abs(steering_angle)

        # If camera not stable, be more conservative with alerts
        if not camera_stable:
            if abs_offset < self.WARNING_ZONE:
                return SafetyLevel.SAFE, "Stabilizing Camera"
            else:
                return SafetyLevel.CAUTION, "Camera Stabilization - Monitor Position"

        # === CONTEXT 1: Sharp steering (lane change or sharp turn) ===
        if abs_steering > 25:
            if abs_offset < self.WARNING_ZONE:
                return SafetyLevel.SAFE, "Sharp Maneuver Detected"
            else:
                return SafetyLevel.CAUTION, "Sharp Turn - Monitor Position"

        # === CONTEXT 2: Moderate steering (gentle turn) ===
        elif abs_steering > 10:
            if abs_offset < self.CAUTION_ZONE:
                return SafetyLevel.SAFE, "Navigating Curve"
            elif abs_offset < self.WARNING_ZONE:
                return SafetyLevel.CAUTION, "Curve - Slight Drift"
            else:
                return SafetyLevel.WARNING, "Excessive Drift in Turn"

        # === CONTEXT 3: Straight driving (strict monitoring) ===
        else:
            if abs_offset < self.SAFE_ZONE:
                return SafetyLevel.SAFE, "Well Centered"
            
            elif abs_offset < self.CAUTION_ZONE:
                if sustained:
                    return SafetyLevel.WARNING, "Sustained Minor Drift"
                return SafetyLevel.SAFE, "Good Lane Position"
            
            elif abs_offset < self.WARNING_ZONE:
                if sustained:
                    return SafetyLevel.DANGER, "SUSTAINED DRIFT - Correct Now"
                return SafetyLevel.WARNING, "Lane Drift Detected"
            
            else:
                return SafetyLevel.DANGER, "LANE DEPARTURE - Immediate Action Required"

    def draw_text_shadow(self, img, text, pos, font, scale, color, thickness):
        """Draw text with shadow"""
        x, y = pos
        cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def visualize_analysis(self, frame, analysis: SafetyAnalysis):
        """
        IMPROVED: Clean, stable visualization
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # === COLOR SCHEME ===
        if analysis.safety_level == SafetyLevel.DANGER:
            alert_color = (0, 50, 255)
            bg_color = (0, 30, 150)
            status_icon = "!"
        elif analysis.safety_level == SafetyLevel.WARNING:
            alert_color = (0, 150, 255)
            bg_color = (0, 90, 150)
            status_icon = "!"
        elif analysis.safety_level == SafetyLevel.CAUTION:
            alert_color = (0, 200, 255)
            bg_color = (0, 120, 150)
            status_icon = "i"
        else:  # SAFE
            alert_color = (0, 255, 100)
            bg_color = (0, 150, 50)
            status_icon = "OK"

        # === TOP ALERT BAR ===
        overlay = vis.copy()
        alert_h = 90
        cv2.rectangle(overlay, (0, 0), (w, alert_h), bg_color, -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

        # Main status message (centered, clean)
        font = cv2.FONT_HERSHEY_DUPLEX
        status_text = f"{analysis.alert_message}"
        msg_size = cv2.getTextSize(status_text, font, 1.1, 3)[0]
        msg_x = (w - msg_size[0]) // 2
        
        self.draw_text_shadow(vis, status_text, (msg_x, 50), font, 1.1, (255, 255, 255), 3)

        # Position info (clean, aligned)
        position_info = f"{analysis.position_text} | {analysis.drift_direction}"
        cv2.putText(vis, position_info, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (220, 220, 220), 2, cv2.LINE_AA)

        # === METRICS PANEL (RIGHT) ===
        panel_overlay = vis.copy()
        panel_w = 400
        panel_h = 300
        panel_x = w - panel_w - 20
        panel_y = 110
        
        cv2.rectangle(panel_overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        vis = cv2.addWeighted(panel_overlay, 0.85, vis, 0.15, 0)
        
        cv2.rectangle(vis, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)

        # Panel title
        cv2.putText(vis, "SAFETY METRICS", (panel_x + 20, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

        px = panel_x + 25
        py = panel_y + 75
        spacing = 48

        info_font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Distance from center
        cv2.putText(vis, "Distance from Center:", (px, py), 
                   info_font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        value_text = f"{abs(analysis.offset_cm):.0f} cm {analysis.position_text}"
        cv2.putText(vis, value_text, (px + 10, py + 26), 
                   info_font, 0.7, alert_color, 2, cv2.LINE_AA)
        py += spacing + 15

        # 2. Vehicle speed
        cv2.putText(vis, "Vehicle Speed:", (px, py), 
                   info_font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{analysis.speed_kmh:.0f} km/h", (px + 10, py + 26),
                   info_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        py += spacing + 15

        # 3. Drift rate
        drift_color = (0, 200, 255) if abs(analysis.drift_rate_cm_s) > 5 else (200, 200, 200)
        cv2.putText(vis, "Drift Rate:", (px, py), 
                   info_font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{analysis.drift_rate_cm_s:+.1f} cm/s", (px + 10, py + 26),
                   info_font, 0.7, drift_color, 2, cv2.LINE_AA)
        py += spacing + 15

        # 4. Detection quality
        qual_color = (0, 255, 100) if analysis.detection_quality_percent > 70 else (0, 200, 255)
        cv2.putText(vis, "Detection Quality:", (px, py), 
                   info_font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{analysis.detection_quality_percent:.0f}%", (px + 10, py + 26),
                   info_font, 0.7, qual_color, 2, cv2.LINE_AA)

        # === BOTTOM STATUS ===
        bottom_y = h - 25

        # Camera stability
        if analysis.camera_stable:
            stable_text = "CAMERA STABLE"
            stable_color = (0, 255, 100)
        else:
            stable_text = "STABILIZING CAMERA"
            stable_color = (0, 200, 255)
        
        cv2.putText(vis, stable_text, (30, bottom_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stable_color, 2, cv2.LINE_AA)

        # Sustained drift warning
        if analysis.sustained_drift:
            warning_text = "PROLONGED DRIFT - ATTENTION REQUIRED"
            warn_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            warn_x = (w - warn_size[0]) // 2
            cv2.putText(vis, warning_text, (warn_x, bottom_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 255), 2, cv2.LINE_AA)

        # === TRAJECTORY GRAPH ===
        if len(analysis.offset_history) > 10:
            graph_overlay = vis.copy()
            graph_w = 280
            graph_h = 90
            graph_x = 30
            graph_y = h - graph_h - 60
            
            cv2.rectangle(graph_overlay, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (25, 25, 25), -1)
            vis = cv2.addWeighted(graph_overlay, 0.85, vis, 0.15, 0)
            
            cv2.rectangle(vis, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (80, 80, 80), 2)
            
            # Title
            cv2.putText(vis, "Position History", (graph_x + 8, graph_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Plot
            history = analysis.offset_history[-60:]
            if len(history) > 1:
                points = []
                for i, val in enumerate(history):
                    x = graph_x + 10 + int((i / len(history)) * (graph_w - 20))
                    y = graph_y + graph_h // 2 - int(val * 40)
                    y = max(graph_y + 5, min(graph_y + graph_h - 5, y))
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(vis, points[i], points[i + 1], alert_color, 2, cv2.LINE_AA)
            
            # Center line
            center_y = graph_y + graph_h // 2
            cv2.line(vis, (graph_x + 10, center_y), 
                    (graph_x + graph_w - 10, center_y), (100, 100, 100), 1)

        return vis


def process_image(image_path, output_path, offset=0.3, speed_kmh=80, steering=0):
    """Process a single image"""
    monitor = SafetyMonitor()
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    vehicle_state = VehicleState(speed_kmh=speed_kmh, steering_angle_deg=steering)
    
    # Simulate frames for realistic smoothing
    for i in range(25):
        simulated_offset = offset + np.random.randn() * 0.02
        analysis = monitor.analyze_driving_safety(
            simulated_offset, 3.7, vehicle_state, 0.9
        )
    
    result = monitor.visualize_analysis(frame, analysis)
    cv2.imwrite(output_path, result)
    
    print(f"Image processed: {output_path}")
    print(f"Safety: {analysis.safety_level.value} | Offset: {abs(analysis.offset_cm):.0f} cm")


def process_video(video_path, output_path, speed_kmh=80):
    """Process video"""
    monitor = SafetyMonitor()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    vehicle_state = VehicleState(speed_kmh=speed_kmh, steering_angle_deg=0.0)
    
    print(f"Processing video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate varying conditions
        base_offset = 0.25 * np.sin(frame_count / 60)
        
        # Add occasional shake
        if frame_count % 150 < 10:
            shake = np.random.randn() * 0.15
        else:
            shake = np.random.randn() * 0.02
        
        simulated_offset = base_offset + shake
        
        # Simulate steering
        if 200 < frame_count % 600 < 300:
            vehicle_state.steering_angle_deg = 15 * np.sin((frame_count % 600 - 200) / 10)
        else:
            vehicle_state.steering_angle_deg = 0
        
        analysis = monitor.analyze_driving_safety(
            simulated_offset, 3.7, vehicle_state, 0.85 + np.random.randn() * 0.05
        )
        
        result = monitor.visualize_analysis(frame, analysis)
        out.write(result)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | Status: {analysis.safety_level.value}")
    
    cap.release()
    out.release()
    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python module3_deviation.py <input> <output>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            speed = int(sys.argv[3]) if len(sys.argv) > 3 else 80
            process_video(input_path, output_path, speed)
        else:
            offset = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
            speed = int(sys.argv[4]) if len(sys.argv) > 4 else 80
            steering = float(sys.argv[5]) if len(sys.argv) > 5 else 0
            process_image(input_path, output_path, offset, speed, steering)