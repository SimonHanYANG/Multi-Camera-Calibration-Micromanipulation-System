import os
import queue
import logging
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QPoint
from stage import Stage

class StageThread(QThread):
    status_signal = pyqtSignal(str, str)  # message, type (info, warning, error)
    calibration_point_signal_image_pixel_stage_calibration = pyqtSignal(int, QPoint)  # Point index, point position
    calibration_complete_signal_image_pixel_stage_calibration = pyqtSignal(bool)  # Calibration completed successfully
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.stage = None
        self.command_queue = queue.Queue()
        self.calibration_file_image_pixel_stage_calibration = "stage_calibration_image_pixel_stage.txt"
        self.calibration_points_image_pixel_stage_calibration = []  # List of image points for calibration
        self.is_calibrated_image_pixel_stage_calibration = False
        self.transformation_matrix_image_pixel_stage_calibration = None
        self.reference_point_image_pixel_stage_calibration = None  # Reference point for movement
        self.logger = logging.getLogger(__name__)
        self.image_width = 1600  # Default values, will be updated
        self.image_height = 1200
        
        # Default pixel to micron conversion
        self.pixel_to_micron_ratio = 0.23125  # Default value for 20x
        
        # Predefined calibration point positions (relative to image center)
        self.calibration_offsets_image_pixel_stage_calibration = [
            (-798, -598),  # Top-left
            (798, -598),   # Top-right
            (798, 598),    # Bottom-right
            (-798, 598),   # Bottom-left
        ]
        
    def set_pixel_to_micron_ratio(self, ratio):
        """Set the pixel to micron conversion ratio"""
        self.pixel_to_micron_ratio = ratio
        self.logger.info(f"Pixel to micron ratio set to: {ratio}")
        
    def set_image_dimensions(self, width, height):
        """Set the current image dimensions for calibration point calculation"""
        self.image_width = width
        self.image_height = height
        
    def run(self):
        self.running = True
        
        try:
            # Initialize stage
            self.stage = Stage()
            self.status_signal.emit("Stage connected successfully", "info")
            
            # Load calibration if exists
            self.load_calibration_image_pixel_stage_calibration()
            
            while self.running:
                try:
                    # Get command from queue with timeout
                    cmd, args = self.command_queue.get(timeout=0.5)
                    
                    if cmd == "move_to_image_pixel_stage_calibration":
                        x, y = args
                        self.move_to_image_point_image_pixel_stage_calibration(x, y)
                    
                    elif cmd == "calibrate_image_pixel_stage_calibration":
                        self.start_calibration_image_pixel_stage_calibration()
                    
                    elif cmd == "add_calibration_point_image_pixel_stage_calibration":
                        img_point, index = args
                        self.add_calibration_point_image_pixel_stage_calibration(img_point, index)
                    
                    elif cmd == "set_image_dimensions":
                        width, height = args
                        self.set_image_dimensions(width, height)
                    
                    elif cmd == "set_reference_point_image_pixel_stage_calibration":
                        x, y = args
                        self.set_reference_point_image_pixel_stage_calibration(x, y)
                    
                    elif cmd == "move_z_relative":
                        z_delta = args
                        self.move_z_relative(z_delta)
                    
                    elif cmd == "set_pixel_to_micron_ratio":
                        ratio = args
                        self.set_pixel_to_micron_ratio(ratio)
                    
                    elif cmd == "get_stage_position":
                        self.get_stage_position()
                    
                    self.command_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            error_msg = f"Stage error: {str(e)}"
            self.logger.error(error_msg)
            self.status_signal.emit(error_msg, "error")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up stage resources"""
        try:
            if hasattr(self, 'stage') and self.stage:
                self.stage.close()
                self.logger.info("Stage closed")
        except Exception as e:
            self.logger.error(f"Stage cleanup error: {e}")
    
    def get_stage_position(self):
        """Get current stage position and emit it"""
        try:
            xy_pos = self.stage.get_xy_position()
            z_pos = self.stage.get_z_position()
            self.status_signal.emit(f"Stage position: X={xy_pos[0]:.2f}, Y={xy_pos[1]:.2f}, Z={z_pos:.2f} μm", "info")
            return xy_pos[0], xy_pos[1], z_pos
        except Exception as e:
            self.logger.error(f"Error getting stage position: {e}")
            self.status_signal.emit(f"Error getting stage position: {str(e)}", "error")
            return None
    
    def move_to_image_point_image_pixel_stage_calibration(self, img_x, img_y):
        """Move stage to position that brings the clicked point to the reference point"""
        if not self.is_calibrated_image_pixel_stage_calibration:
            self.status_signal.emit("Stage not calibrated. Please calibrate first.", "warning")
            return
            
        if self.reference_point_image_pixel_stage_calibration is None:
            self.status_signal.emit("Reference point not set. Please set a reference point first.", "warning")
            return
        
        try:
            # Get current stage position
            current_stage_pos = self.stage.get_xy_position()
            
            # Calculate the offset between clicked point and reference point in image coordinates
            dx_img = img_x - self.reference_point_image_pixel_stage_calibration[0]
            dy_img = img_y - self.reference_point_image_pixel_stage_calibration[1]
            
            # Convert image offset to stage offset using the transformation matrix
            dx_stage = dx_img * self.transformation_matrix_image_pixel_stage_calibration['x_scale']
            dy_stage = dy_img * self.transformation_matrix_image_pixel_stage_calibration['y_scale']
            
            # Calculate the new stage position (move in opposite direction to bring clicked point to reference)
            new_stage_x = current_stage_pos[0] + dx_stage * -1
            new_stage_y = current_stage_pos[1] + dy_stage
            
            # Move the stage
            self.stage.move_xy_to_absolute(new_stage_x, new_stage_y)
            
            self.status_signal.emit(f"Moving to position: ({new_stage_x:.2f}, {new_stage_y:.2f})", "info")
            
        except Exception as e:
            self.logger.error(f"Move error: {e}")
            self.status_signal.emit(f"Move failed: {str(e)}", "error")
    
    def move_z_relative(self, z_delta):
        """Move Z stage by relative amount"""
        try:
            # Get current Z position for logging
            current_z = self.stage.get_z_position()
            
            # Move Z stage
            self.stage.move_z_relative(z_delta)
            
            # Get new Z position
            new_z = self.stage.get_z_position()
            
            self.status_signal.emit(f"Z moved from {current_z:.2f} to {new_z:.2f} (Δz: {z_delta})", "info")
            
        except Exception as e:
            self.logger.error(f"Z move error: {e}")
            self.status_signal.emit(f"Z move failed: {str(e)}", "error")
    
    def set_reference_point_image_pixel_stage_calibration(self, x, y):
        """Set the reference point for stage movement"""
        if not self.is_calibrated_image_pixel_stage_calibration:
            self.status_signal.emit("Cannot set reference point: Stage not calibrated. Please calibrate first.", "warning")
            return False
        
        self.reference_point_image_pixel_stage_calibration = (x, y)
        self.status_signal.emit(f"Reference point set to ({x}, {y})", "info")
        self.save_calibration_image_pixel_stage_calibration()  # Save the updated reference point
        return True
    
    def start_calibration_image_pixel_stage_calibration(self):
        """Start the calibration process"""
        self.calibration_points_image_pixel_stage_calibration = []
        self.is_calibrated_image_pixel_stage_calibration = False
        self.reference_point_image_pixel_stage_calibration = None  # Clear reference point
        
        # Request first calibration point
        self.request_next_calibration_point_image_pixel_stage_calibration()
    
    def request_next_calibration_point_image_pixel_stage_calibration(self):
        """Request the next calibration point"""
        point_index = len(self.calibration_points_image_pixel_stage_calibration)
        
        if point_index >= len(self.calibration_offsets_image_pixel_stage_calibration):
            # All points collected, complete calibration
            self.complete_calibration_image_pixel_stage_calibration()
            return
        
        # Calculate the next point position based on image center and offset
        img_center_x = self.image_width // 2
        img_center_y = self.image_height // 2
        
        # Get the offset for this calibration point
        offset_x, offset_y = self.calibration_offsets_image_pixel_stage_calibration[point_index]
        
        # Calculate the point position
        point_x = img_center_x + offset_x
        point_y = img_center_y + offset_y
        
        # Send signal to display the point
        self.calibration_point_signal_image_pixel_stage_calibration.emit(point_index, QPoint(point_x, point_y))
        
        # Update status
        point_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.status_signal.emit(f"Please click on the {point_names[point_index]} calibration point (red marker)", "warning")
    
    def add_calibration_point_image_pixel_stage_calibration(self, img_point, index):
        """Add a calibration point"""
        self.calibration_points_image_pixel_stage_calibration.append((img_point.x(), img_point.y()))
        
        self.logger.info(f"Added calibration point {index+1}: Image({img_point.x()}, {img_point.y()})")
        
        # Request next calibration point
        self.request_next_calibration_point_image_pixel_stage_calibration()
    
    def complete_calibration_image_pixel_stage_calibration(self):
        """Complete the calibration process"""
        if len(self.calibration_points_image_pixel_stage_calibration) < 4:
            self.status_signal.emit("Calibration failed: Need at least 4 points.", "error")
            self.calibration_complete_signal_image_pixel_stage_calibration.emit(False)
            return
            
        try:
            # Calculate transformation matrix
            self.calculate_transformation_image_pixel_stage_calibration()
            
            # Set default reference point (center of calibration points)
            x_sum = sum(p[0] for p in self.calibration_points_image_pixel_stage_calibration)
            y_sum = sum(p[1] for p in self.calibration_points_image_pixel_stage_calibration)
            self.reference_point_image_pixel_stage_calibration = (x_sum / len(self.calibration_points_image_pixel_stage_calibration), 
                                                                  y_sum / len(self.calibration_points_image_pixel_stage_calibration))
            
            # Save calibration
            self.save_calibration_image_pixel_stage_calibration()
            
            self.is_calibrated_image_pixel_stage_calibration = True
            self.status_signal.emit("Calibration completed successfully. Please set a reference point.", "info")
            self.calibration_complete_signal_image_pixel_stage_calibration.emit(True)
            
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            self.status_signal.emit(f"Calibration failed: {str(e)}", "error")
            self.calibration_complete_signal_image_pixel_stage_calibration.emit(False)
    
    def calculate_transformation_image_pixel_stage_calibration(self):
        """Calculate the transformation matrix from image to stage coordinates"""
        # Calculate the scale factors using the pixel to micron ratio
        x_scale_avg = self.pixel_to_micron_ratio
        y_scale_avg = self.pixel_to_micron_ratio
        
        # Store the transformation matrix
        self.transformation_matrix_image_pixel_stage_calibration = {
            'x_scale': x_scale_avg,
            'y_scale': y_scale_avg
        }
        
        self.logger.info(f"Transformation matrix: x_scale={x_scale_avg:.6f}, y_scale={y_scale_avg:.6f}")
    
    def save_calibration_image_pixel_stage_calibration(self):
        """Save calibration data to file"""
        try:
            with open(self.calibration_file_image_pixel_stage_calibration, 'w') as f:
                # Save transformation matrix
                f.write(f"{self.transformation_matrix_image_pixel_stage_calibration['x_scale']},{self.transformation_matrix_image_pixel_stage_calibration['y_scale']}\n")
                
                # Save reference point (if set)
                if self.reference_point_image_pixel_stage_calibration:
                    f.write(f"{self.reference_point_image_pixel_stage_calibration[0]},{self.reference_point_image_pixel_stage_calibration[1]}\n")
                else:
                    f.write("None,None\n")
                
                # Save calibration points
                for point in self.calibration_points_image_pixel_stage_calibration:
                    f.write(f"{point[0]},{point[1]}\n")
                    
                # Save pixel to micron ratio
                f.write(f"pixel_to_micron_ratio:{self.pixel_to_micron_ratio}\n")
                    
            self.logger.info(f"Calibration saved to {self.calibration_file_image_pixel_stage_calibration}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
    
    def load_calibration_image_pixel_stage_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(self.calibration_file_image_pixel_stage_calibration):
            self.logger.info("No calibration file found.")
            return
            
        try:
            with open(self.calibration_file_image_pixel_stage_calibration, 'r') as f:
                lines = f.readlines()
                
                if len(lines) < 6:  # Need at least transformation matrix, reference point, and 4 calibration points
                    self.logger.error("Invalid calibration file format.")
                    return
                
                # Load transformation matrix
                x_scale, y_scale = map(float, lines[0].strip().split(','))
                self.transformation_matrix_image_pixel_stage_calibration = {
                    'x_scale': x_scale,
                    'y_scale': y_scale
                }
                
                # Load reference point
                ref_x_str, ref_y_str = lines[1].strip().split(',')
                if ref_x_str != "None" and ref_y_str != "None":
                    ref_x, ref_y = float(ref_x_str), float(ref_y_str)
                    self.reference_point_image_pixel_stage_calibration = (ref_x, ref_y)
                
                # Load calibration points
                self.calibration_points_image_pixel_stage_calibration = []
                for i in range(2, len(lines)):
                    line = lines[i].strip()
                    if line and not line.startswith("pixel_to_micron_ratio:"):
                        x, y = map(float, line.split(','))
                        self.calibration_points_image_pixel_stage_calibration.append((x, y))
                    elif line.startswith("pixel_to_micron_ratio:"):
                        self.pixel_to_micron_ratio = float(line.split(':')[1])
                
                self.is_calibrated_image_pixel_stage_calibration = True
                self.logger.info("Calibration loaded successfully.")
                status_msg = "Calibration loaded from file."
                if self.reference_point_image_pixel_stage_calibration is None:
                    status_msg += " Reference point not set. Please set a reference point."
                self.status_signal.emit(status_msg, "info")
                
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
    
    def request_move_to_image_pixel_stage_calibration(self, img_x, img_y):
        """Request to move the stage to the given image coordinates"""
        self.command_queue.put(("move_to_image_pixel_stage_calibration", (img_x, img_y)))
    
    def request_move_z_relative(self, z_delta):
        """Request to move the Z stage by a relative amount"""
        self.command_queue.put(("move_z_relative", z_delta))
    
    def request_calibration_image_pixel_stage_calibration(self):
        """Request to start calibration"""
        self.command_queue.put(("calibrate_image_pixel_stage_calibration", None))
    
    def request_add_calibration_point_image_pixel_stage_calibration(self, img_point, index):
        """Request to add a calibration point"""
        self.command_queue.put(("add_calibration_point_image_pixel_stage_calibration", (img_point, index)))
    
    def request_set_image_dimensions(self, width, height):
        """Request to set image dimensions"""
        self.command_queue.put(("set_image_dimensions", (width, height)))
    
    def request_set_reference_point_image_pixel_stage_calibration(self, x, y):
        """Request to set reference point"""
        self.command_queue.put(("set_reference_point_image_pixel_stage_calibration", (x, y)))
    
    def request_set_pixel_to_micron_ratio(self, ratio):
        """Request to set pixel to micron ratio"""
        self.command_queue.put(("set_pixel_to_micron_ratio", ratio))
    
    def request_get_stage_position(self):
        """Request to get current stage position"""
        self.command_queue.put(("get_stage_position", None))
    
    def stop(self):
        self.running = False
        self.wait(1000)