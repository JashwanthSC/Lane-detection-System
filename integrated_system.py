"""
Integrated Lane Detection System - PRODUCTION VERSION
IMPROVEMENTS:
- Clean, stable visualization across all modules
- Robust lane detection with temporal smoothing
- Stable driver alerts (no false triggers)
- Green tracking line positioned ahead of vehicle
- Professional, demo-ready output
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Import all three improved modules
from module1_reliability import LaneReliabilityAssessor, EnvironmentContext
from module2_detection import LaneDetector, Lane
from module3_deviation import SafetyMonitor, SafetyAnalysis, SafetyLevel, VehicleState


def force_size(img, target_size):
    """Resize image to exact target size"""
    w, h = target_size
    if img.shape[1] != w or img.shape[0] != h:
        return cv2.resize(img, (w, h))
    return img


class IntegratedLaneSystem:
    """
    Complete integrated system with:
    - Environmental assessment
    - Robust lane detection
    - Stable safety monitoring
    """
    def __init__(self, calibrate=False):
        self.module1 = LaneReliabilityAssessor()
        self.module2 = LaneDetector(calibrate=calibrate)
        self.module3 = SafetyMonitor()
        self.vehicle_state = VehicleState(speed_kmh=80.0, steering_angle_deg=0.0)
        self.reliability_threshold = 0.3

    def process_frame(self, frame):
        """Process frame through all 3 modules"""
        # Module 1: Environment assessment
        environment = self.module1.assess_frame(frame)
        
        # Module 2: Lane detection (only if environment is suitable)
        if environment.reliability_score >= self.reliability_threshold:
            lane = self.module2.detect_lanes(frame)
        else:
            lane = Lane(None, None, 0, 0, 0, 0, 0)
        
        # Module 3: Safety analysis
        if lane.left_fit is not None and lane.right_fit is not None:
            safety = self.module3.analyze_driving_safety(
                raw_offset=lane.center_offset,
                lane_width=lane.lane_width,
                vehicle_state=self.vehicle_state,
                detection_confidence=lane.confidence
            )
        else:
            # Create default safety analysis when no lanes detected
            safety = SafetyAnalysis(
                offset_cm=0.0,
                position_text="Unknown",
                drift_rate_cm_s=0.0,
                drift_direction="Unknown",
                safety_level=SafetyLevel.WARNING,
                alert_message="Lane Detection Lost",
                sustained_drift=False,
                camera_stable=False,
                detection_quality_percent=0.0,
                speed_kmh=self.vehicle_state.speed_kmh,
                steering_angle_deg=self.vehicle_state.steering_angle_deg,
                lane_width_m=0.0,
                offset_history=[]
            )
        
        return {"environment": environment, "lane": lane, "safety": safety}
    
    def visualize(self, frame, results):
        """Create visualizations for all modules"""
        v1 = self.module1.visualize_assessment(frame.copy(), results["environment"])
        v2 = self.module2.visualize_lanes(frame.copy(), results["lane"])
        v3 = self.module3.visualize_analysis(frame.copy(), results["safety"])
        return v1, v2, v3


def process_video_all_modules(video_path, output_base="output"):
    """
    Process video through all modules
    
    Input:  input/test_video.mp4
    Output: output/module1/test_video_module1.mp4
            output/module2/test_video_module2.mp4
            output/module3/test_video_module3.mp4
            output/combined/test_video_full_system.mp4
    """
    system = IntegratedLaneSystem()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    base_name = Path(video_path).stem
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Create output folders
    os.makedirs(f"{output_base}/module1", exist_ok=True)
    os.makedirs(f"{output_base}/module2", exist_ok=True)
    os.makedirs(f"{output_base}/module3", exist_ok=True)
    os.makedirs(f"{output_base}/combined", exist_ok=True)
    
    # Create video writers
    writer_m1 = cv2.VideoWriter(
        f"{output_base}/module1/{base_name}_module1.mp4",
        fourcc, fps, (1280, 720)
    )
    writer_m2 = cv2.VideoWriter(
        f"{output_base}/module2/{base_name}_module2.mp4",
        fourcc, fps, (1280, 720)
    )
    writer_m3 = cv2.VideoWriter(
        f"{output_base}/module3/{base_name}_module3.mp4",
        fourcc, fps, (1280, 720)
    )
    writer_combined = cv2.VideoWriter(
        f"{output_base}/combined/{base_name}_full_system.mp4",
        fourcc, fps, (1280, 2160)
    )
    
    print("=" * 70)
    print(f"PROCESSING: {video_path}")
    print(f"Frames: {total} | FPS: {fps} | Size: {w}x{h}")
    print("=" * 70)
    
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Simulate realistic steering variations
            if count % 300 < 100:
                system.vehicle_state.steering_angle_deg = 12 * np.sin(count / 30)
            else:
                system.vehicle_state.steering_angle_deg = 0
            
            # Process frame
            results = system.process_frame(frame)
            v1, v2, v3 = system.visualize(frame, results)
            
            # Write to individual module videos
            writer_m1.write(force_size(v1, (1280, 720)))
            writer_m2.write(force_size(v2, (1280, 720)))
            writer_m3.write(force_size(v3, (1280, 720)))
            
            # Write combined (stacked vertically)
            v1_sized = force_size(v1, (1280, 720))
            v2_sized = force_size(v2, (1280, 720))
            v3_sized = force_size(v3, (1280, 720))
            combined = np.vstack([v1_sized, v2_sized, v3_sized])
            writer_combined.write(combined)
            
            count += 1
            if count % 30 == 0:
                env_st = results["environment"].confidence_flag
                lane_st = "Detected" if results["lane"].left_fit is not None else "Lost"
                safe_st = results["safety"].safety_level.value
                progress = (count / total * 100) if total > 0 else 0
                print(f"  Frame {count}/{total} ({progress:.0f}%) | "
                      f"Env:{env_st} | Lane:{lane_st} | Safety:{safe_st}")
        
        except Exception as e:
            print(f"  WARNING: Frame {count}: {e}")
            count += 1
            continue
    
    # Release everything
    cap.release()
    writer_m1.release()
    writer_m2.release()
    writer_m3.release()
    writer_combined.release()
    
    print("=" * 70)
    print(f"COMPLETE! Processed {count} frames")
    print(f"\nOutput files:")
    print(f"  {output_base}/module1/{base_name}_module1.mp4")
    print(f"  {output_base}/module2/{base_name}_module2.mp4")
    print(f"  {output_base}/module3/{base_name}_module3.mp4")
    print(f"  {output_base}/combined/{base_name}_full_system.mp4")
    print("=" * 70)


def process_image_all_modules(image_path, output_base="output"):
    """
    Process image through all modules
    
    Input:  input/test.jpg
    Output: output/module1/test_module1.jpg
            output/module2/test_module2.jpg
            output/module3/test_module3.jpg
            output/combined/test_full.jpg
    """
    system = IntegratedLaneSystem()
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"ERROR: Could not load: {image_path}")
        return
    
    print(f"Processing: {image_path}")
    
    results = system.process_frame(frame)
    v1, v2, v3 = system.visualize(frame, results)
    
    base_name = Path(image_path).stem
    
    # Create output folders
    os.makedirs(f"{output_base}/module1", exist_ok=True)
    os.makedirs(f"{output_base}/module2", exist_ok=True)
    os.makedirs(f"{output_base}/module3", exist_ok=True)
    os.makedirs(f"{output_base}/combined", exist_ok=True)
    
    # Save individual outputs
    cv2.imwrite(f"{output_base}/module1/{base_name}_module1.jpg", v1)
    cv2.imwrite(f"{output_base}/module2/{base_name}_module2.jpg", v2)
    cv2.imwrite(f"{output_base}/module3/{base_name}_module3.jpg", v3)
    
    # Save combined
    v1_r = cv2.resize(v1, (1280, 720))
    v2_r = cv2.resize(v2, (1280, 720))
    v3_r = cv2.resize(v3, (1280, 720))
    combined = np.vstack([v1_r, v2_r, v3_r])
    cv2.imwrite(f"{output_base}/combined/{base_name}_full.jpg", combined)
    
    # Print results
    env = results["environment"]
    lane = results["lane"]
    safety = results["safety"]
    
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"Module 1: {env.reliability_score:.0%} ({env.confidence_flag})")
    print(f"Module 2: {'Lanes Detected' if lane.left_fit is not None else 'No Lanes'}")
    print(f"Module 3: {safety.safety_level.value}")
    print(f"\nOutput files:")
    print(f"  {output_base}/module1/{base_name}_module1.jpg")
    print(f"  {output_base}/module2/{base_name}_module2.jpg")
    print(f"  {output_base}/module3/{base_name}_module3.jpg")
    print(f"  {output_base}/combined/{base_name}_full.jpg")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("INTEGRATED LANE DETECTION SYSTEM")
        print("=" * 70)
        print("\nUsage:")
        print("  python integrated_system.py <input_file>")
        print("\nExamples:")
        print("  python integrated_system.py input/test_video.mp4")
        print("  python integrated_system.py input/test.jpg")
        print("=" * 70)
    else:
        input_path = sys.argv[1]
        
        if not os.path.exists(input_path):
            print(f"ERROR: File not found: {input_path}")
        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            process_video_all_modules(input_path)
        else:
            process_image_all_modules(input_path)