import os
import time
import logging
import numpy as np
import cv2
from scipy.optimize import least_squares
from PyQt6.QtCore import QThread, pyqtSignal, QPoint

class ConcentricCircleThread(QThread):
    status_signal = pyqtSignal(str, str)  # message, type
    request_point_signal = pyqtSignal(str)  # request message
    drawing_complete_signal = pyqtSignal(np.ndarray, list, list)  # image_with_curves, curve1_points, curve2_points
    
    def __init__(self, stage_thread):
        super().__init__()
        self.stage_thread = stage_thread
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # State tracking
        self.current_step = "none"  # "center", "curve1", "curve2", "complete"
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
        
        # Fitted conic parameters
        self.curve1_conic = None
        self.curve2_conic = None
        
    def get_pixel_to_micron_ratio(self):
        """Get the current pixel to micron ratio from stage thread"""
        try:
            # Get the ratio from stage thread which should be updated from UI
            if hasattr(self.stage_thread, 'pixel_to_micron_ratio'):
                ratio = self.stage_thread.pixel_to_micron_ratio
                self.logger.info(f"Using pixel to micron ratio: {ratio}")
                return ratio
            else:
                # Fallback to default value
                default_ratio = 0.23125
                self.logger.warning(f"Pixel to micron ratio not found in stage thread, using default: {default_ratio}")
                return default_ratio
        except Exception as e:
            self.logger.error(f"Error getting pixel to micron ratio: {e}")
            return 0.23125  # Default fallback
        
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
            
            # Also log the current pixel to micron ratio being used
            current_ratio = self.get_pixel_to_micron_ratio()
            self.logger.info(f"Stage origin position set to: {self.stage_origin_position}")
            self.logger.info(f"Using pixel to micron ratio: {current_ratio}")
            
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
        self.curve1_conic = None
        self.curve2_conic = None
        
    def add_point(self, point):
        """Add a point based on current step"""
        if self.current_step == "center":
            self.center_point = point
            self.current_step = "curve1"
            self.collected_points = 0
            self.request_point_signal.emit(f"圆心已设置，请依次点击第一个二次曲线上的点 (1/{self.points_needed})")
            
        elif self.current_step == "curve1":
            self.curve1_points.append(point)
            self.collected_points += 1
            
            if self.collected_points < self.points_needed:
                self.request_point_signal.emit(f"请继续点击第一个二次曲线上的点 ({self.collected_points + 1}/{self.points_needed})")
            else:
                # All 5 points collected for curve 1, automatically proceed to curve 2
                try:
                    self.curve1_conic = self.fit_conic(self.curve1_points)
                    if not self.check_conic_validity(self.curve1_conic, self.current_image.shape):
                        self.status_signal.emit("第一个二次曲线拟合失败或超出图像边界，请重新选择点", "warning")
                        self.curve1_points = []
                        self.collected_points = 0
                        self.curve1_conic = None
                        self.request_point_signal.emit(f"请重新点击第一个二次曲线上的点 (1/{self.points_needed})")
                        return
                    
                    # Move to curve 2
                    self.current_step = "curve2"
                    self.collected_points = 0
                    self.request_point_signal.emit(f"第一个二次曲线完成，请依次点击第二个二次曲线上的点 (1/{self.points_needed})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to fit first conic: {e}")
                    self.status_signal.emit("第一个二次曲线拟合失败，请重新选择点", "error")
                    self.curve1_points = []
                    self.collected_points = 0
                    self.curve1_conic = None
                    self.request_point_signal.emit(f"请重新点击第一个二次曲线上的点 (1/{self.points_needed})")
                    
        elif self.current_step == "curve2":
            self.curve2_points.append(point)
            self.collected_points += 1
            
            if self.collected_points < self.points_needed:
                self.request_point_signal.emit(f"请继续点击第二个二次曲线上的点 ({self.collected_points + 1}/{self.points_needed})")
            else:
                # All 5 points collected for curve 2, automatically complete
                try:
                    self.curve2_conic = self.fit_conic(self.curve2_points)
                    if not self.check_conic_validity(self.curve2_conic, self.current_image.shape):
                        self.status_signal.emit("第二个二次曲线拟合失败或超出图像边界，请重新选择点", "warning")
                        self.curve2_points = []
                        self.collected_points = 0
                        self.curve2_conic = None
                        self.request_point_signal.emit(f"请重新点击第二个二次曲线上的点 (1/{self.points_needed})")
                        return
                    
                    # Complete drawing automatically
                    self.complete_drawing()
                    
                except Exception as e:
                    self.logger.error(f"Failed to fit second conic: {e}")
                    self.status_signal.emit("第二个二次曲线拟合失败，请重新选择点", "error")
                    self.curve2_points = []
                    self.collected_points = 0
                    self.curve2_conic = None
                    self.request_point_signal.emit(f"请重新点击第二个二次曲线上的点 (1/{self.points_needed})")
    
    def fit_conic(self, points):
        """
        Fit a conic section (ellipse) to the given points using least squares
        Returns conic matrix representation: [A B C D E F] where
        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        if len(points) < 5:
            raise ValueError("Need at least 5 points to fit a conic")
        
        # Convert points to numpy arrays
        n = len(points)
        x = np.array([p.x() for p in points])
        y = np.array([p.y() for p in points])
        
        # Construct design matrix for conic fitting
        # Conic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        D = np.column_stack([
            x**2,      # A coefficient
            x * y,     # B coefficient  
            y**2,      # C coefficient
            x,         # D coefficient
            y,         # E coefficient
            np.ones(n) # F coefficient
        ])
        
        # For ellipse constraint (B^2 - 4AC < 0), we use constrained fitting
        # First try direct algebraic fitting with constraint
        try:
            # Use constraint that coefficient C = 1 (normalize by C)
            # This gives us: Ax^2 + Bxy + y^2 + Dx + Ey + F = 0
            D_constrained = np.column_stack([
                x**2,      # A coefficient
                x * y,     # B coefficient
                x,         # D coefficient
                y,         # E coefficient
                np.ones(n) # F coefficient
            ])
            
            # Right hand side is -y^2
            rhs = -y**2
            
            # Solve least squares: D_constrained * params = rhs
            params, residuals, rank, s = np.linalg.lstsq(D_constrained, rhs, rcond=None)
            
            # Reconstruct full conic parameters [A B C D E F]
            A, B, D_coef, E, F = params
            C = 1.0
            
            conic_params = np.array([A, B, C, D_coef, E, F])
            
            # Check if it's an ellipse (B^2 - 4AC < 0)
            discriminant = B**2 - 4*A*C
            if discriminant >= 0:
                # Not an ellipse, try general conic fitting
                return self.fit_general_conic(points)
            
            return conic_params
            
        except Exception as e:
            self.logger.warning(f"Constrained conic fitting failed: {e}, trying general fitting")
            return self.fit_general_conic(points)
    
    def fit_general_conic(self, points):
        """Fit general conic using SVD"""
        n = len(points)
        x = np.array([p.x() for p in points])
        y = np.array([p.y() for p in points])
        
        # Design matrix
        D = np.column_stack([
            x**2, x*y, y**2, x, y, np.ones(n)
        ])
        
        # Use SVD to find the least squares solution
        U, S, Vt = np.linalg.svd(D)
        
        # The solution is the last column of V (last row of Vt)
        conic_params = Vt[-1, :]
        
        # Normalize so that the largest coefficient is reasonable
        max_coef = np.max(np.abs(conic_params[:3]))  # Only consider quadratic terms
        if max_coef > 0:
            conic_params = conic_params / max_coef
            
        return conic_params
    
    def conic_to_matrix(self, conic_params):
        """Convert conic parameters to matrix form"""
        A, B, C, D, E, F = conic_params
        
        matrix = np.array([
            [A,   B/2, D/2],
            [B/2, C,   E/2],
            [D/2, E/2, F  ]
        ])
        
        return matrix
    
    def check_conic_validity(self, conic_params, image_shape):
        """Check if the fitted conic is valid and within image bounds"""
        if conic_params is None:
            return False
            
        try:
            A, B, C, D, E, F = conic_params
            
            # Check if coefficients are reasonable (not too large or too small)
            if np.any(np.abs(conic_params) > 1e6) or np.any(np.abs(conic_params[:3]) < 1e-10):
                return False
            
            # For ellipse: B^2 - 4AC should be negative
            discriminant = B**2 - 4*A*C
            if discriminant >= 0:
                self.logger.warning(f"Conic is not an ellipse, discriminant: {discriminant}")
                # Still allow it, might be acceptable for calibration
            
            # Check if conic intersects image bounds reasonably
            height, width = image_shape[:2]
            
            # Sample points along image boundary and check if conic passes through reasonable region
            test_points = []
            # Top and bottom edges
            for x in np.linspace(0, width-1, 20):
                test_points.extend([(x, 0), (x, height-1)])
            # Left and right edges  
            for y in np.linspace(0, height-1, 20):
                test_points.extend([(0, y), (width-1, y)])
            
            # Evaluate conic at test points
            inside_count = 0
            for x, y in test_points:
                value = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
                if abs(value) < width * height:  # Reasonable scale
                    inside_count += 1
            
            # If most test points give reasonable values, consider it valid
            return inside_count > len(test_points) * 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking conic validity: {e}")
            return False
    
    def draw_conic(self, image, conic_params, color, thickness=1):
        """Draw the fitted conic on the image"""
        if conic_params is None:
            return
            
        try:
            A, B, C, D, E, F = conic_params
            height, width = image.shape[:2]
            
            # Generate points on the conic by parametric approach
            points = []
            
            # For each x coordinate, solve for y
            for x in range(0, width, 2):  # Sample every 2 pixels for performance
                # Solve: Cy^2 + (Bx + E)y + (Ax^2 + Dx + F) = 0
                a_coef = C
                b_coef = B*x + E
                c_coef = A*x**2 + D*x + F
                
                if abs(a_coef) > 1e-10:  # Avoid division by zero
                    discriminant = b_coef**2 - 4*a_coef*c_coef
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        y1 = (-b_coef + sqrt_disc) / (2*a_coef)
                        y2 = (-b_coef - sqrt_disc) / (2*a_coef)
                        
                        # Add points if they're within image bounds
                        if 0 <= y1 < height:
                            points.append([x, int(y1)])
                        if 0 <= y2 < height and abs(y2 - y1) > 1:
                            points.append([x, int(y2)])
            
            # Also solve for x given y to get more complete curve
            for y in range(0, height, 2):
                # Solve: Ax^2 + (By + D)x + (Cy^2 + Ey + F) = 0
                a_coef = A
                b_coef = B*y + D
                c_coef = C*y**2 + E*y + F
                
                if abs(a_coef) > 1e-10:
                    discriminant = b_coef**2 - 4*a_coef*c_coef
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        x1 = (-b_coef + sqrt_disc) / (2*a_coef)
                        x2 = (-b_coef - sqrt_disc) / (2*a_coef)
                        
                        if 0 <= x1 < width:
                            points.append([int(x1), y])
                        if 0 <= x2 < width and abs(x2 - x1) > 1:
                            points.append([int(x2), y])
            
            # Remove duplicates and sort points
            if points:
                points = np.array(points)
                unique_points = np.unique(points, axis=0)
                
                if len(unique_points) > 3:
                    # Sort points to create a connected curve
                    # Find convex hull or sort by angle from center
                    center_x = np.mean(unique_points[:, 0])
                    center_y = np.mean(unique_points[:, 1])
                    
                    angles = np.arctan2(unique_points[:, 1] - center_y, 
                                      unique_points[:, 0] - center_x)
                    sorted_indices = np.argsort(angles)
                    sorted_points = unique_points[sorted_indices]
                    
                    # Draw the curve
                    cv2.polylines(image, [sorted_points], True, color, thickness)
                    
        except Exception as e:
            self.logger.error(f"Error drawing conic: {e}")
    
    def image_to_stage_coordinates(self, img_point):
        """Convert image coordinates to stage coordinates based on image top-left as origin"""
        if self.stage_origin_position is None:
            return (0.0, 0.0, 0.0)
        
        # Calculate offset from image top-left corner in pixels
        pixel_x = img_point.x()
        pixel_y = img_point.y()
        
        # Get current pixel to micron ratio from system settings
        pixel_to_micron = self.get_pixel_to_micron_ratio()
        micron_x = pixel_x * pixel_to_micron
        micron_y = pixel_y * pixel_to_micron
        
        # Calculate stage coordinates (image top-left is the origin)
        stage_x = self.stage_origin_position[0] + micron_x
        stage_y = self.stage_origin_position[1] + micron_y
        stage_z = self.stage_origin_position[2]  # Z position remains the same
        
        return (stage_x, stage_y, stage_z)
    
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
            
            # Draw curve points with green color for both curves
            for point in self.curve1_points:
                cv2.circle(result_image, (point.x(), point.y()), 3, self.curve1_color, -1)
                cv2.line(result_image, (point.x() - 5, point.y()), (point.x() + 5, point.y()), self.curve1_color, 1)
                cv2.line(result_image, (point.x(), point.y() - 5), (point.x(), point.y() + 5), self.curve1_color, 1)
            
            for point in self.curve2_points:
                cv2.circle(result_image, (point.x(), point.y()), 3, self.curve1_color, -1)
                cv2.line(result_image, (point.x() - 5, point.y()), (point.x() + 5, point.y()), self.curve1_color, 1)
                cv2.line(result_image, (point.x(), point.y() - 5), (point.x(), point.y() + 5), self.curve1_color, 1)
            
            # Draw fitted conics
            self.draw_conic(result_image, self.curve1_conic, self.curve1_color, 2)
            self.draw_conic(result_image, self.curve2_conic, self.curve2_color, 2)
            
            # Save results
            self.save_results(result_image)
            
            # Emit completion signal
            self.drawing_complete_signal.emit(result_image, self.curve1_points, self.curve2_points)
            self.current_step = "complete"
            self.status_signal.emit("同心二次曲线拟合完成并已自动保存", "info")
            
        except Exception as e:
            self.logger.error(f"Error completing drawing: {e}")
            self.status_signal.emit(f"绘制完成时出错: {str(e)}", "error")
    
    def save_results(self, result_image):
        """Save the results to files"""
        try:
            # Create directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"Concentric_Circle_Calibration/{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Get current pixel to micron ratio for saving
            current_pixel_to_micron = self.get_pixel_to_micron_ratio()
            
            # Save image
            image_filename = os.path.join(save_dir, f"concentric_conics_{timestamp}.png")
            cv2.imwrite(image_filename, result_image)
            
            # Save center point coordinates
            center_stage_coords = self.image_to_stage_coordinates(self.center_point)
            
            # Save curve 1 data
            curve1_rgb = f"{self.curve1_color[2]:03d}_{self.curve1_color[1]:03d}_{self.curve1_color[0]:03d}"
            curve1_filename = os.path.join(save_dir, f"{curve1_rgb}_{timestamp}.txt")
            
            with open(curve1_filename, 'w') as f:
                f.write("# Concentric Conic 1 Data for Camera Calibration\n")
                f.write("# Based on 'The Common Self-polar Triangle of Concentric Circles' (Huang et al., CVPR 2015)\n")
                f.write("# Image origin (top-left) corresponds to stage position:\n")
                f.write(f"# Stage Origin: X={self.stage_origin_position[0]:.3f}, Y={self.stage_origin_position[1]:.3f}, Z={self.stage_origin_position[2]:.3f} μm\n")
                f.write(f"# Pixel to Micron Ratio: {current_pixel_to_micron:.6f} μm/pixel\n")
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
                
                f.write("\n# Conic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0\n")
                if self.curve1_conic is not None:
                    A, B, C, D, E, F = self.curve1_conic
                    f.write(f"A: {A:.6f}\n")
                    f.write(f"B: {B:.6f}\n")
                    f.write(f"C: {C:.6f}\n")
                    f.write(f"D: {D:.6f}\n")
                    f.write(f"E: {E:.6f}\n")
                    f.write(f"F: {F:.6f}\n")
                    
                    # Save conic matrix for calibration
                    f.write("\n# Conic Matrix (3x3) for calibration:\n")
                    conic_matrix = self.conic_to_matrix(self.curve1_conic)
                    for row in conic_matrix:
                        f.write(f"# [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
            
            # Save curve 2 data
            curve2_rgb = f"{self.curve2_color[2]:03d}_{self.curve2_color[1]:03d}_{self.curve2_color[0]:03d}"
            curve2_filename = os.path.join(save_dir, f"{curve2_rgb}_{timestamp}.txt")
            
            with open(curve2_filename, 'w') as f:
                f.write("# Concentric Conic 2 Data for Camera Calibration\n")
                f.write("# Based on 'The Common Self-polar Triangle of Concentric Circles' (Huang et al., CVPR 2015)\n")
                f.write("# Image origin (top-left) corresponds to stage position:\n")
                f.write(f"# Stage Origin: X={self.stage_origin_position[0]:.3f}, Y={self.stage_origin_position[1]:.3f}, Z={self.stage_origin_position[2]:.3f} μm\n")
                f.write(f"# Pixel to Micron Ratio: {current_pixel_to_micron:.6f} μm/pixel\n")
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
                
                f.write("\n# Conic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0\n")
                if self.curve2_conic is not None:
                    A, B, C, D, E, F = self.curve2_conic
                    f.write(f"A: {A:.6f}\n")
                    f.write(f"B: {B:.6f}\n")
                    f.write(f"C: {C:.6f}\n")
                    f.write(f"D: {D:.6f}\n")
                    f.write(f"E: {E:.6f}\n")
                    f.write(f"F: {F:.6f}\n")
                    
                    # Save conic matrix for calibration
                    f.write("\n# Conic Matrix (3x3) for calibration:\n")
                    conic_matrix = self.conic_to_matrix(self.curve2_conic)
                    for row in conic_matrix:
                        f.write(f"# [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
            
            # Save calibration data file
            calib_filename = os.path.join(save_dir, f"calibration_data_{timestamp}.txt")
            with open(calib_filename, 'w') as f:
                f.write("# Camera Calibration Data - Concentric Conics\n")
                f.write("# Compatible with Huang et al. CVPR 2015 algorithm\n")
                f.write("# Use this data with camera calibration software\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Image dimensions: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n")
                f.write(f"# Pixel to micron ratio: {current_pixel_to_micron:.6f} μm/pixel\n")
                f.write(f"# Stage origin position: X={self.stage_origin_position[0]:.3f}, Y={self.stage_origin_position[1]:.3f}, Z={self.stage_origin_position[2]:.3f} μm\n")
                f.write("# \n")
                
                # Write both conic matrices in format suitable for MATLAB/OpenCV
                if self.curve1_conic is not None and self.curve2_conic is not None:
                    f.write("# Conic 1 Matrix:\n")
                    conic1_matrix = self.conic_to_matrix(self.curve1_conic)
                    for i, row in enumerate(conic1_matrix):
                        f.write(f"C1_{i+1}: {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
                    
                    f.write("# \n# Conic 2 Matrix:\n")
                    conic2_matrix = self.conic_to_matrix(self.curve2_conic)
                    for i, row in enumerate(conic2_matrix):
                        f.write(f"C2_{i+1}: {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
                    
                    # Additional calibration parameters
                    f.write("# \n# Additional Parameters for Calibration:\n")
                    f.write(f"PixelToMicronRatio: {current_pixel_to_micron:.8f}\n")
                    f.write(f"ImageWidth: {self.current_image.shape[1]}\n")
                    f.write(f"ImageHeight: {self.current_image.shape[0]}\n")
                    f.write(f"CenterX_Image: {self.center_point.x()}\n")
                    f.write(f"CenterY_Image: {self.center_point.y()}\n")
                    f.write(f"CenterX_Stage: {center_stage_coords[0]:.3f}\n")
                    f.write(f"CenterY_Stage: {center_stage_coords[1]:.3f}\n")
                    f.write(f"CenterZ_Stage: {center_stage_coords[2]:.3f}\n")
            
            self.logger.info(f"Calibration data saved to {save_dir}")
            self.logger.info(f"Used pixel to micron ratio: {current_pixel_to_micron}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.status_signal.emit(f"保存结果时出错: {str(e)}", "error")
    
    def stop(self):
        self.running = False
        self.wait(1000)