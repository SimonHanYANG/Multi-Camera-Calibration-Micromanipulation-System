import os
import time
import logging
import numpy as np
import cv2
from scipy.optimize import curve_fit
from PyQt6.QtCore import QThread, pyqtSignal, QPoint

class ConcentricCircleThread(QThread):
    status_signal = pyqtSignal(str, str)  # message, type
    request_point_signal = pyqtSignal(str)  # request message
    drawing_complete_signal = pyqtSignal(np.ndarray, list, list)  # image_with_curves, curve1_points, curve2_points
    request_confirm_signal = pyqtSignal(str)  # request confirmation message
    
    def __init__(self, stage_thread):
        super().__init__()
        self.stage_thread = stage_thread
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # State tracking
        self.current_step = "none"  # "center", "curve1", "curve1_confirm", "curve2", "curve2_confirm", "complete"
        self.center_point = None
        self.curve1_points = []
        self.curve2_points = []
        self.collected_points = 0
        self.points_needed = 5
        
        # Colors for curves (BGR format for OpenCV)
        self.curve1_color = (0, 255, 0)    # Green (RGB: 0, 255, 0)
        self.curve2_color = (255, 0, 0)    # Blue (RGB: 0, 0, 255)
        
        # Current image and stage origin position
        self.current_image = None
        self.stage_origin_position = None  # Stage position corresponding to image top-left corner
        
    def start_drawing(self, image):
        """Start the concentric circle drawing process"""
        self.current_image = image.copy()
        self.reset_state()
        
        # Get current stage position as the origin (corresponding to image top-left corner)
        try:
            current_pos = self.stage_thread.get_stage_position()
            if current_pos is None:
                self.status_signal.emit("无法获取载物台位置", "error")
                return
            self.stage_origin_position = current_pos
            self.logger.info(f"Stage origin position set to: {self.stage_origin_position}")
        except Exception as e:
            self.logger.error(f"Error getting stage position: {e}")
            self.status_signal.emit(f"获取载物台位置失败: {str(e)}", "error")
            return
        
        self.current_step = "center"
        self.request_point_signal.emit("请点击圆心位置")
        
    def reset_state(self):
        """Reset all state variables"""
        self.center_point = None
        self.curve1_points = []
        self.curve2_points = []
        self.collected_points = 0
        self.current_step = "none"
        self.stage_origin_position = None
        
    def add_point(self, point):
        """Add a point based on current step"""
        if self.current_step == "center":
            self.center_point = point
            self.current_step = "curve1"
            self.collected_points = 0
            self.request_point_signal.emit(f"请依次点击第一个二次曲线上的点 (1/{self.points_needed})")
            
        elif self.current_step == "curve1":
            self.curve1_points.append(point)
            self.collected_points += 1
            
            if self.collected_points < self.points_needed:
                self.request_point_signal.emit(f"请继续点击第一个二次曲线上的点 ({self.collected_points + 1}/{self.points_needed})")
            else:
                # All 5 points collected for curve 1, ask for confirmation
                self.current_step = "curve1_confirm"
                self.request_confirm_signal.emit("第一个二次曲线的5个点已选择完成，请点击'完成'按钮开始拟合第一个曲线")
                
        elif self.current_step == "curve2":
            self.curve2_points.append(point)
            self.collected_points += 1
            
            if self.collected_points < self.points_needed:
                self.request_point_signal.emit(f"请继续点击第二个二次曲线上的点 ({self.collected_points + 1}/{self.points_needed})")
            else:
                # All 5 points collected for curve 2, ask for confirmation
                self.current_step = "curve2_confirm"
                self.request_confirm_signal.emit("第二个二次曲线的5个点已选择完成，请点击'完成'按钮开始拟合第二个曲线")
    
    def confirm_curve_fitting(self):
        """Confirm and proceed with curve fitting"""
        if self.current_step == "curve1_confirm":
            # Fit first curve
            try:
                curve1_params = self.fit_quadratic_curve(self.curve1_points)
                if not self.check_curve_in_bounds(curve1_params, self.current_image.shape):
                    self.status_signal.emit("第一个二次曲线超出图像边界，请重新选择点", "warning")
                    self.curve1_points = []
                    self.collected_points = 0
                    self.current_step = "curve1"
                    self.request_point_signal.emit(f"请重新点击第一个二次曲线上的点 (1/{self.points_needed})")
                    return
                
                self.status_signal.emit("第一个二次曲线拟合完成", "info")
                self.current_step = "curve2"
                self.collected_points = 0
                self.request_point_signal.emit(f"请依次点击第二个二次曲线上的点 (1/{self.points_needed})")
                
            except Exception as e:
                self.logger.error(f"Failed to fit first curve: {e}")
                self.status_signal.emit("第一个二次曲线拟合失败，请重新选择点", "error")
                self.curve1_points = []
                self.collected_points = 0
                self.current_step = "curve1"
                self.request_point_signal.emit(f"请重新点击第一个二次曲线上的点 (1/{self.points_needed})")
                
        elif self.current_step == "curve2_confirm":
            # Fit second curve and complete
            try:
                curve2_params = self.fit_quadratic_curve(self.curve2_points)
                if not self.check_curve_in_bounds(curve2_params, self.current_image.shape):
                    self.status_signal.emit("第二个二次曲线超出图像边界，请重新选择点", "warning")
                    self.curve2_points = []
                    self.collected_points = 0
                    self.current_step = "curve2"
                    self.request_point_signal.emit(f"请重新点击第二个二次曲线上的点 (1/{self.points_needed})")
                    return
                
                self.status_signal.emit("第二个二次曲线拟合完成", "info")
                self.complete_drawing()
                
            except Exception as e:
                self.logger.error(f"Failed to fit second curve: {e}")
                self.status_signal.emit("第二个二次曲线拟合失败，请重新选择点", "error")
                self.curve2_points = []
                self.collected_points = 0
                self.current_step = "curve2"
                self.request_point_signal.emit(f"请重新点击第二个二次曲线上的点 (1/{self.points_needed})")
    
    def image_to_stage_coordinates(self, img_point):
        """Convert image coordinates to stage coordinates based on image top-left as origin"""
        if self.stage_origin_position is None:
            return (0.0, 0.0, 0.0)
        
        # Calculate offset from image top-left corner in pixels
        pixel_x = img_point.x()
        pixel_y = img_point.y()
        
        # Convert pixels to microns using the pixel to micron ratio
        pixel_to_micron = getattr(self.stage_thread, 'pixel_to_micron_ratio', 0.23125)
        micron_x = pixel_x * pixel_to_micron
        micron_y = pixel_y * pixel_to_micron
        
        # Calculate stage coordinates (image top-left is the origin)
        stage_x = self.stage_origin_position[0] + micron_x
        stage_y = self.stage_origin_position[1] + micron_y
        stage_z = self.stage_origin_position[2]  # Z position remains the same
        
        return (stage_x, stage_y, stage_z)
    
    def fit_quadratic_curve(self, points):
        """Fit a quadratic curve to the given points"""
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a quadratic curve")
            
        # Convert points to numpy arrays
        x_data = np.array([p.x() for p in points])
        y_data = np.array([p.y() for p in points])
        
        # Define quadratic function: y = ax^2 + bx + c
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        # Fit the curve
        popt, _ = curve_fit(quadratic, x_data, y_data)
        
        return popt  # Returns [a, b, c]
    
    def check_curve_in_bounds(self, curve_params, image_shape):
        """Check if the fitted curve stays within image bounds"""
        a, b, c = curve_params
        height, width = image_shape[:2]
        
        # Check curve at multiple x positions across the image width
        x_positions = np.linspace(0, width - 1, 100)
        y_positions = a * x_positions**2 + b * x_positions + c
        
        # Check if all y positions are within bounds
        return np.all((y_positions >= 0) & (y_positions < height))
    
    def draw_curve(self, image, curve_params, color, thickness=1):
        """Draw the fitted curve on the image"""
        a, b, c = curve_params
        height, width = image.shape[:2]
        
        # Generate points along the curve
        x_positions = np.arange(0, width)
        y_positions = a * x_positions**2 + b * x_positions + c
        
        # Filter points that are within image bounds
        valid_mask = (y_positions >= 0) & (y_positions < height)
        valid_x = x_positions[valid_mask]
        valid_y = y_positions[valid_mask]
        
        # Draw the curve
        if len(valid_x) > 1:
            points = np.column_stack((valid_x, valid_y)).astype(np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(image, [points], False, color, thickness)
    
    def complete_drawing(self):
        """Complete the drawing process and save results"""
        try:
            # Create image with curves
            result_image = self.current_image.copy()
            
            # Convert to BGR if grayscale
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            # Draw center point (red circle with cross)
            center = self.center_point
            cv2.circle(result_image, (center.x(), center.y()), 5, (0, 0, 255), -1)  # Red circle
            cv2.line(result_image, (center.x() - 10, center.y()), (center.x() + 10, center.y()), (0, 0, 255), 2)
            cv2.line(result_image, (center.x(), center.y() - 10), (center.x(), center.y() + 10), (0, 0, 255), 2)
            
            # Draw curve points with green color for curve 1
            for point in self.curve1_points:
                cv2.circle(result_image, (point.x(), point.y()), 3, self.curve1_color, -1)
                cv2.line(result_image, (point.x() - 5, point.y()), (point.x() + 5, point.y()), self.curve1_color, 1)
                cv2.line(result_image, (point.x(), point.y() - 5), (point.x(), point.y() + 5), self.curve1_color, 1)
            
            # Draw curve points with blue color for curve 2
            for point in self.curve2_points:
                cv2.circle(result_image, (point.x(), point.y()), 3, self.curve2_color, -1)
                cv2.line(result_image, (point.x() - 5, point.y()), (point.x() + 5, point.y()), self.curve2_color, 1)
                cv2.line(result_image, (point.x(), point.y() - 5), (point.x(), point.y() + 5), self.curve2_color, 1)
            
            # Fit and draw curves
            curve1_params = self.fit_quadratic_curve(self.curve1_points)
            curve2_params = self.fit_quadratic_curve(self.curve2_points)
            
            self.draw_curve(result_image, curve1_params, self.curve1_color, 1)
            self.draw_curve(result_image, curve2_params, self.curve2_color, 1)
            
            # Save results
            self.save_results(result_image, curve1_params, curve2_params)
            
            # Emit completion signal
            self.drawing_complete_signal.emit(result_image, self.curve1_points, self.curve2_points)
            self.current_step = "complete"
            self.status_signal.emit("同心圆绘制完成并保存", "info")
            
        except Exception as e:
            self.logger.error(f"Error completing drawing: {e}")
            self.status_signal.emit(f"绘制完成时出错: {str(e)}", "error")
    
    def save_results(self, result_image, curve1_params, curve2_params):
        """Save the results to files"""
        try:
            # Create directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"Concentric_Circle_Calibration/{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save image
            image_filename = os.path.join(save_dir, f"concentric_circles_{timestamp}.png")
            cv2.imwrite(image_filename, result_image)
            
            # Save center point coordinates
            center_stage_coords = self.image_to_stage_coordinates(self.center_point)
            
            # Save curve 1 data
            curve1_rgb = f"{self.curve1_color[2]:03d}_{self.curve1_color[1]:03d}_{self.curve1_color[0]:03d}"  # BGR to RGB
            curve1_filename = os.path.join(save_dir, f"{curve1_rgb}_{timestamp}.txt")
            
            with open(curve1_filename, 'w') as f:
                f.write("# Curve 1 Data\n")
                f.write("# Image origin (top-left) corresponds to stage position:\n")
                f.write(f"# Stage Origin: X={self.stage_origin_position[0]:.3f}, Y={self.stage_origin_position[1]:.3f}, Z={self.stage_origin_position[2]:.3f} μm\n")
                f.write("# \n")
                f.write("# Center point coordinates:\n")
                f.write(f"Center_Image: {self.center_point.x()}, {self.center_point.y()}\n")
                f.write(f"Center_Stage: {center_stage_coords[0]:.3f}, {center_stage_coords[1]:.3f}, {center_stage_coords[2]:.3f}\n")
                f.write("# \n")
                f.write("# Point coordinates (Image coordinates | Stage coordinates in μm)\n")
                for i, point in enumerate(self.curve1_points):
                    stage_coords = self.image_to_stage_coordinates(point)
                    f.write(f"Point_{i+1}_Image: {point.x()}, {point.y()}\n")
                    f.write(f"Point_{i+1}_Stage: {stage_coords[0]:.3f}, {stage_coords[1]:.3f}, {stage_coords[2]:.3f}\n")
                
                f.write("\n# Quadratic curve equation: y = ax^2 + bx + c (in image coordinates)\n")
                f.write(f"a: {curve1_params[0]:.6f}\n")
                f.write(f"b: {curve1_params[1]:.6f}\n")
                f.write(f"c: {curve1_params[2]:.6f}\n")
            
            # Save curve 2 data
            curve2_rgb = f"{self.curve2_color[2]:03d}_{self.curve2_color[1]:03d}_{self.curve2_color[0]:03d}"  # BGR to RGB
            curve2_filename = os.path.join(save_dir, f"{curve2_rgb}_{timestamp}.txt")
            
            with open(curve2_filename, 'w') as f:
                f.write("# Curve 2 Data\n")
                f.write("# Image origin (top-left) corresponds to stage position:\n")
                f.write(f"# Stage Origin: X={self.stage_origin_position[0]:.3f}, Y={self.stage_origin_position[1]:.3f}, Z={self.stage_origin_position[2]:.3f} μm\n")
                f.write("# \n")
                f.write("# Center point coordinates:\n")
                f.write(f"Center_Image: {self.center_point.x()}, {self.center_point.y()}\n")
                f.write(f"Center_Stage: {center_stage_coords[0]:.3f}, {center_stage_coords[1]:.3f}, {center_stage_coords[2]:.3f}\n")
                f.write("# \n")
                f.write("# Point coordinates (Image coordinates | Stage coordinates in μm)\n")
                for i, point in enumerate(self.curve2_points):
                    stage_coords = self.image_to_stage_coordinates(point)
                    f.write(f"Point_{i+1}_Image: {point.x()}, {point.y()}\n")
                    f.write(f"Point_{i+1}_Stage: {stage_coords[0]:.3f}, {stage_coords[1]:.3f}, {stage_coords[2]:.3f}\n")
                
                f.write("\n# Quadratic curve equation: y = ax^2 + bx + c (in image coordinates)\n")
                f.write(f"a: {curve2_params[0]:.6f}\n")
                f.write(f"b: {curve2_params[1]:.6f}\n")
                f.write(f"c: {curve2_params[2]:.6f}\n")
            
            self.logger.info(f"Results saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.status_signal.emit(f"保存结果时出错: {str(e)}", "error")
    
    def stop(self):
        self.running = False
        self.wait(1000)