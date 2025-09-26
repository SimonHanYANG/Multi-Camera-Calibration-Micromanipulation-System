import sys
import os
import time
import logging
import numpy as np
import cv2
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                            QPushButton, QHBoxLayout, QMessageBox, QGridLayout, QGroupBox, 
                            QLineEdit, QFormLayout, QFileDialog)

# Import custom threads
from camera_thread import CameraThread
from video_recording_thread import VideoRecordingThread
from stage_thread import StageThread
from concentric_circle_thread import ConcentricCircleThread

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robotic Micromanipulation Multi-cam Calibration system")
        
        # Set up the UI
        self._setup_ui()
        
        # Initialize threads
        self._setup_threads()
        
        # State variables
        self.calibration_mode_image_pixel_stage_calibration = False
        self.calibration_point_index_image_pixel_stage_calibration = -1
        self.current_calibration_point_image_pixel_stage_calibration = None
        self.reference_point_mode_image_pixel_stage_calibration = False
        
        # Visual feedback markers
        self.target_marker = None
        self.reference_marker_image_pixel_stage_calibration = None
        self.marker_timer = QTimer()
        self.marker_timer.timeout.connect(self.clear_markers)
        
        # Z-axis movement step size
        self.z_step = 5.0  # 5 μm per scroll step
        
        # Recording state
        self.is_recording = False
        self.current_image_size = (1600, 1200)  # Default size
        
        # Camera/File switch state
        self.camera_mode = True  # True for camera, False for file
        self.current_file_image = None
        
        # Concentric circle drawing state
        self.concentric_circle_mode = False
        self.current_displayed_image = None
        self.concentric_circle_points = []  # Store clicked points for visual feedback
        
        # Check camera availability and start appropriate mode
        self.check_and_start_camera()
        
        # Start other threads
        self.start_other_threads()
        
        # Set focus to image label to receive key events
        self.image_label.setFocus()
    
    def _setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left side: Image display
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel, 1)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image_label.setMinimumSize(640, 480)
        self.left_layout.addWidget(self.image_label)
        
        # Status labels
        self.status_label = QLabel("Starting... Double-click on image to move stage (requires calibration)")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        self.status_label.setFixedHeight(30)
        self.left_layout.addWidget(self.status_label)
        
        # Right side: Control panel
        self.right_panel = QWidget()
        self.right_panel.setFixedWidth(350)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.main_layout.addWidget(self.right_panel, 0)
        
        # Camera/File Control group
        self.camera_file_group = QGroupBox("Camera/File Control")
        self.camera_file_layout = QVBoxLayout()
        self.camera_file_group.setLayout(self.camera_file_layout)
        
        # Camera/File mode switch button
        self.camera_file_switch_btn = QPushButton("Switch to File Mode")
        self.camera_file_switch_btn.setFixedHeight(40)
        self.camera_file_switch_btn.clicked.connect(self.toggle_camera_file_mode)
        self.camera_file_layout.addWidget(self.camera_file_switch_btn)
        
        # File load button (only enabled in file mode)
        self.file_load_btn = QPushButton("Load Image File")
        self.file_load_btn.setFixedHeight(40)
        self.file_load_btn.setStyleSheet("background-color: #E0E0E0;")
        self.file_load_btn.setEnabled(False)  # Initially disabled
        self.file_load_btn.clicked.connect(self.load_image_file_dialog)
        self.camera_file_layout.addWidget(self.file_load_btn)
        
        # Video Recording group
        self.recording_group = QGroupBox("Video Recording")
        self.recording_layout = QVBoxLayout()
        self.recording_group.setLayout(self.recording_layout)
        
        recording_btn_layout = QHBoxLayout()
        
        self.start_recording_btn = QPushButton("Start Recording")
        self.start_recording_btn.setFixedHeight(40)
        self.start_recording_btn.setStyleSheet("background-color: #99FF99;")
        self.start_recording_btn.clicked.connect(self.start_recording)
        recording_btn_layout.addWidget(self.start_recording_btn)
        
        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setFixedHeight(40)
        self.stop_recording_btn.setStyleSheet("background-color: #FF9999;")
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        recording_btn_layout.addWidget(self.stop_recording_btn)
        
        self.recording_layout.addLayout(recording_btn_layout)
        
        self.recording_status_label = QLabel("Not recording")
        self.recording_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status_label.setStyleSheet("background-color: #F0F0F0; padding: 5px;")
        self.recording_layout.addWidget(self.recording_status_label)
        
        # Pixel to Micron Conversion group
        self.conversion_group = QGroupBox("Pixel to Micron Conversion")
        self.conversion_layout = QFormLayout()
        self.conversion_group.setLayout(self.conversion_layout)
        
        self.pixel_micron_input = QLineEdit()
        self.pixel_micron_input.setText("0.23125")  # Default for 20x
        self.pixel_micron_input.setPlaceholderText("Enter μm per pixel")
        self.conversion_layout.addRow("μm per pixel:", self.pixel_micron_input)
        
        self.set_conversion_btn = QPushButton("Set Conversion Ratio")
        self.set_conversion_btn.clicked.connect(self.set_pixel_to_micron_ratio)
        self.conversion_layout.addWidget(self.set_conversion_btn)
        
        # Stage control group
        self.stage_group = QGroupBox("Stage Control")
        self.stage_layout = QVBoxLayout()
        self.stage_group.setLayout(self.stage_layout)
        
        self.calibrate_btn_image_pixel_stage_calibration = QPushButton("Calibrate Stage (Image-Pixel-Stage)")
        self.calibrate_btn_image_pixel_stage_calibration.setFixedHeight(40)
        self.calibrate_btn_image_pixel_stage_calibration.clicked.connect(self.start_calibration_image_pixel_stage_calibration)
        self.stage_layout.addWidget(self.calibrate_btn_image_pixel_stage_calibration)
        
        self.set_ref_point_btn_image_pixel_stage_calibration = QPushButton("Set Reference Point (Image-Pixel-Stage)")
        self.set_ref_point_btn_image_pixel_stage_calibration.setFixedHeight(40)
        self.set_ref_point_btn_image_pixel_stage_calibration.clicked.connect(self.start_set_reference_point_image_pixel_stage_calibration)
        self.stage_layout.addWidget(self.set_ref_point_btn_image_pixel_stage_calibration)
        
        z_info_label = QLabel("Use the mouse wheel to control the Z-axis")
        z_info_label.setStyleSheet("color: #0066CC; font-style: italic;")
        z_info_label.setWordWrap(True)
        self.stage_layout.addWidget(z_info_label)
        
        # Concentric Circle Drawing group
        self.circle_group = QGroupBox("Concentric Quadratic Curve Drawing")
        self.circle_layout = QVBoxLayout()
        self.circle_group.setLayout(self.circle_layout)
        
        self.draw_circles_btn = QPushButton("Draw Concentric Quadratic Curve")
        self.draw_circles_btn.setFixedHeight(40)
        self.draw_circles_btn.setStyleSheet("background-color: #FFE6CC;")
        self.draw_circles_btn.clicked.connect(self.start_concentric_circle_drawing)
        self.circle_layout.addWidget(self.draw_circles_btn)
        
        # Add groups to control panel
        self.right_layout.addWidget(self.camera_file_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.recording_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.conversion_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.stage_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.circle_group)
        self.right_layout.addStretch(1)
        
        # Connect mouse events
        self.image_label.mousePressEvent = self.image_click
        self.image_label.mouseDoubleClickEvent = self.image_double_click
        self.image_label.wheelEvent = self.image_wheel_event
    
    def _setup_threads(self):
        # Create threads
        self.camera_thread = CameraThread()
        self.stage_thread = StageThread()
        self.video_recording_thread = VideoRecordingThread()
        self.concentric_circle_thread = ConcentricCircleThread(self.stage_thread)
        
        # Connect signals
        self.camera_thread.new_image_signal.connect(self.update_camera_display)
        self.camera_thread.new_image_signal.connect(self.video_recording_thread.update_frame)
        self.camera_thread.error_signal.connect(lambda msg: self.show_error(msg))
        
        self.stage_thread.status_signal.connect(self.update_status)
        self.stage_thread.calibration_point_signal_image_pixel_stage_calibration.connect(self.show_calibration_point_image_pixel_stage_calibration)
        self.stage_thread.calibration_complete_signal_image_pixel_stage_calibration.connect(self.on_calibration_complete_image_pixel_stage_calibration)
        self.stage_thread.position_signal.connect(self.on_stage_position_received)
        
        self.video_recording_thread.status_signal.connect(self.update_recording_status)
        self.video_recording_thread.recording_started_signal.connect(self.on_recording_started)
        self.video_recording_thread.recording_stopped_signal.connect(self.on_recording_stopped)
        
        self.concentric_circle_thread.status_signal.connect(self.update_status)
        self.concentric_circle_thread.request_point_signal.connect(self.show_circle_drawing_request)
        self.concentric_circle_thread.drawing_complete_signal.connect(self.on_circle_drawing_complete)
    
    def check_and_start_camera(self):
        """Check camera availability and start appropriate mode"""
        if self.camera_thread.check_camera_availability():
            self.camera_mode = True
            self.camera_file_switch_btn.setText("Switch to File Mode")
            self.file_load_btn.setEnabled(False)
            self.file_load_btn.setStyleSheet("background-color: #E0E0E0;")
            self.update_status("Camera detected, starting camera feed", "info")
        else:
            self.camera_mode = False
            self.camera_file_switch_btn.setText("Switch to Camera Mode")
            self.file_load_btn.setEnabled(True)
            self.file_load_btn.setStyleSheet("background-color: #CCE5FF;")
            self.show_no_camera_message()
            self.update_status("No camera detected, please load an image file", "warning")
    
    def show_no_camera_message(self):
        """Show 'Camera not connected' message"""
        # Create a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder.fill(50)  # Dark gray background
        
        # Convert to QPixmap and add text
        h, w, ch = placeholder.shape
        bytes_per_line = ch * w
        q_image = QImage(placeholder.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Draw text
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)
        
        text = "Camera is not connected"
        text_rect = painter.fontMetrics().boundingRect(text)
        x = (w - text_rect.width()) // 2
        y = (h - text_rect.height()) // 2
        painter.drawText(x, y, text)
        painter.end()
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
    
    def start_other_threads(self):
        """Start non-camera threads"""
        self.stage_thread.start()
        self.video_recording_thread.start()
        self.concentric_circle_thread.start()
        
        # Start camera thread if camera is available
        if self.camera_mode and self.camera_thread.camera_available:
            self.camera_thread.start()
    
    def toggle_camera_file_mode(self):
        """Toggle between camera and file mode"""
        if self.camera_mode:
            # Switch to file mode
            self.camera_mode = False
            self.camera_file_switch_btn.setText("Switch to Camera Mode")
            self.file_load_btn.setEnabled(True)
            self.file_load_btn.setStyleSheet("background-color: #CCE5FF;")
            self.update_status("Switched to file mode. Click 'Load Image File' to load an image.", "info")
            
            # Show placeholder if no file is loaded
            if self.current_file_image is None:
                self.show_file_mode_message()
        else:
            # Switch back to camera mode
            if self.camera_thread.camera_available:
                self.camera_mode = True
                self.camera_file_switch_btn.setText("Switch to File Mode")
                self.file_load_btn.setEnabled(False)
                self.file_load_btn.setStyleSheet("background-color: #E0E0E0;")
                self.current_file_image = None
                self.update_status("Switched back to camera mode", "info")
            else:
                self.update_status("No camera available, cannot switch to camera mode", "warning")
    
    def show_file_mode_message(self):
        """Show 'File mode' message when no file is loaded"""
        # Create a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder.fill(70)  # Slightly lighter gray
        
        # Convert to QPixmap and add text
        h, w, ch = placeholder.shape
        bytes_per_line = ch * w
        q_image = QImage(placeholder.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Draw text
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        
        text = "File Mode - Load an image"
        text_rect = painter.fontMetrics().boundingRect(text)
        x = (w - text_rect.width()) // 2
        y = (h - text_rect.height()) // 2
        painter.drawText(x, y, text)
        painter.end()
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
    
    def load_image_file_dialog(self):
        """Open file dialog to load an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image File", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            self.load_image_file(file_path)
    
    def load_image_file(self, file_path):
        """Load and display an image file"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_file_image = image
            self.update_file_display(image)
            self.update_status(f"Loaded image: {os.path.basename(file_path)}", "info")
            
        except Exception as e:
            self.show_error(f"Failed to load image: {str(e)}")
    
    def update_camera_display(self, image):
        """Update display with camera image"""
        if self.camera_mode:
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            self.current_displayed_image = image.copy()
            self.update_display(image)
    
    def update_file_display(self, image):
        """Update display with file image"""
        if not self.camera_mode:
            self.current_displayed_image = image.copy()
            self.update_display(image)
    
    def update_display(self, image):
        """Update the display with the processed image at original size"""
        try:
            h, w = image.shape[:2]
            self.current_image_size = (w, h)
            
            # Update stage thread with current image dimensions
            self.stage_thread.request_set_image_dimensions(w, h)
            
            # Ensure image is RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Assume it's already RGB
                pass
            
            bytes_per_line = 3 * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create pixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            
            # Draw markers if needed
            painter = QPainter(pixmap)
            
            # Draw recording indicator
            if self.is_recording:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(20, 20, 20, 20)
                
                font = QFont()
                font.setPointSize(14)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QColor(255, 0, 0))
                painter.drawText(50, 35, "REC")
            
            # Draw calibration point (red)
            if self.calibration_mode_image_pixel_stage_calibration and self.current_calibration_point_image_pixel_stage_calibration:
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawEllipse(self.current_calibration_point_image_pixel_stage_calibration, 10, 10)
                painter.drawLine(self.current_calibration_point_image_pixel_stage_calibration.x() - 15, self.current_calibration_point_image_pixel_stage_calibration.y(),
                               self.current_calibration_point_image_pixel_stage_calibration.x() + 15, self.current_calibration_point_image_pixel_stage_calibration.y())
                painter.drawLine(self.current_calibration_point_image_pixel_stage_calibration.x(), self.current_calibration_point_image_pixel_stage_calibration.y() - 15,
                               self.current_calibration_point_image_pixel_stage_calibration.x(), self.current_calibration_point_image_pixel_stage_calibration.y() + 15)
            
            # Draw reference point (green)
            if self.reference_marker_image_pixel_stage_calibration:
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.drawEllipse(self.reference_marker_image_pixel_stage_calibration, 12, 12)
                painter.drawLine(self.reference_marker_image_pixel_stage_calibration.x() - 18, self.reference_marker_image_pixel_stage_calibration.y(),
                               self.reference_marker_image_pixel_stage_calibration.x() + 18, self.reference_marker_image_pixel_stage_calibration.y())
                painter.drawLine(self.reference_marker_image_pixel_stage_calibration.x(), self.reference_marker_image_pixel_stage_calibration.y() - 18,
                               self.reference_marker_image_pixel_stage_calibration.x(), self.reference_marker_image_pixel_stage_calibration.y() + 18)
            
            # Draw target marker (blue)
            if self.target_marker:
                painter.setPen(QPen(QColor(0, 120, 255), 3))
                painter.drawEllipse(self.target_marker, 10, 10)
                painter.drawLine(self.target_marker.x() - 15, self.target_marker.y(),
                               self.target_marker.x() + 15, self.target_marker.y())
                painter.drawLine(self.target_marker.x(), self.target_marker.y() - 15,
                               self.target_marker.x(), self.target_marker.y() + 15)
            
            # Draw concentric circle points if in concentric circle mode
            if self.concentric_circle_mode:
                # Draw center point if exists (red circle with cross)
                if hasattr(self.concentric_circle_thread, 'center_point') and self.concentric_circle_thread.center_point:
                    center = self.concentric_circle_thread.center_point
                    painter.setPen(QPen(QColor(255, 0, 0), 3))
                    painter.drawEllipse(center.x() - 5, center.y() - 5, 10, 10)
                    painter.drawLine(center.x() - 10, center.y(), center.x() + 10, center.y())
                    painter.drawLine(center.x(), center.y() - 10, center.x(), center.y() + 10)
                
                # Draw curve 1 points (green)
                for point in self.concentric_circle_thread.curve1_points:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawEllipse(point.x() - 3, point.y() - 3, 6, 6)
                    painter.drawLine(point.x() - 5, point.y(), point.x() + 5, point.y())
                    painter.drawLine(point.x(), point.y() - 5, point.x(), point.y() + 5)
                
                # Draw curve 2 points (green - same color for both curves' points)
                for point in self.concentric_circle_thread.curve2_points:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawEllipse(point.x() - 3, point.y() - 3, 6, 6)
                    painter.drawLine(point.x() - 5, point.y(), point.x() + 5, point.y())
                    painter.drawLine(point.x(), point.y() - 5, point.x(), point.y() + 5)
            
            painter.end()
            
            # Set the pixmap to the label
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.size())
            
            # Adjust window size on first image
            if not hasattr(self, 'window_sized'):
                screen = QApplication.primaryScreen().availableGeometry()
                total_width = min(w + self.right_panel.width() + 50, screen.width() - 100)
                total_height = min(h + self.status_label.height() + 80, screen.height() - 100)
                
                self.resize(total_width, total_height)
                
                frame_geo = self.frameGeometry()
                screen_center = screen.center()
                frame_geo.moveCenter(screen_center)
                self.move(frame_geo.topLeft())
                
                self.window_sized = True
                
        except Exception as e:
            logger.error(f"Display update error: {e}")
            self.show_error(f"Display update error: {e}")
    
    def set_pixel_to_micron_ratio(self):
        """Set the pixel to micron conversion ratio"""
        try:
            ratio = float(self.pixel_micron_input.text())
            if ratio <= 0:
                raise ValueError("Ratio must be positive")
            
            self.stage_thread.request_set_pixel_to_micron_ratio(ratio)
            self.update_status(f"Pixel to micron ratio set to: {ratio}", "info")
            
        except ValueError as e:
            self.show_error(f"Invalid conversion ratio: {str(e)}")
    
    def start_concentric_circle_drawing(self):
        """Start concentric circle drawing process"""
        if self.current_displayed_image is None:
            self.show_error("No image available for drawing")
            return
        
        self.concentric_circle_mode = True
        self.concentric_circle_thread.start_drawing(self.current_displayed_image)
        # Request current stage position for coordinate calculation
        self.stage_thread.request_get_stage_position()
    
    def show_circle_drawing_request(self, message):
        """Show circle drawing request message"""
        self.update_status(message, "warning")
    
    def on_circle_drawing_complete(self, result_image, curve1_points, curve2_points):
        """Handle completion of circle drawing"""
        self.concentric_circle_mode = False
        # Update display with the result image that has the curves drawn
        self.update_display(result_image)
    
    def on_stage_position_received(self, x, y, z):
        """Handle stage position signal"""
        # This is called when stage position is received
        # Can be used for logging or other purposes
        pass
    
    # Event handlers
    def image_click(self, event):
        """Handle mouse click events on the image"""
        pos = event.position()
        click_x = int(pos.x())
        click_y = int(pos.y())
        click_point = QPoint(click_x, click_y)
        
        if self.calibration_mode_image_pixel_stage_calibration:
            # Add calibration point
            self.stage_thread.request_add_calibration_point_image_pixel_stage_calibration(click_point, self.calibration_point_index_image_pixel_stage_calibration)
            self.current_calibration_point_image_pixel_stage_calibration = None
            
        elif self.reference_point_mode_image_pixel_stage_calibration:
            # Set reference point
            self.stage_thread.request_set_reference_point_image_pixel_stage_calibration(click_x, click_y)
            self.reference_marker_image_pixel_stage_calibration = click_point
            self.update_status(f"Reference point set to ({click_x}, {click_y})", "info")
            self.reference_point_mode_image_pixel_stage_calibration = False
            
        elif self.concentric_circle_mode:
            # Add point for concentric circle drawing
            self.concentric_circle_thread.add_point(click_point)
    
    def image_double_click(self, event):
        """Handle mouse double-click events on the image"""
        if self.calibration_mode_image_pixel_stage_calibration or self.reference_point_mode_image_pixel_stage_calibration or self.concentric_circle_mode:
            return
            
        pos = event.position()
        click_x = int(pos.x())
        click_y = int(pos.y())
        click_point = QPoint(click_x, click_y)
        
        if not self.stage_thread.is_calibrated_image_pixel_stage_calibration:
            self.update_status("Stage not calibrated. Please calibrate first.", "warning")
            return
            
        if self.stage_thread.reference_point_image_pixel_stage_calibration is None:
            self.update_status("Reference point not set. Please set reference point first.", "warning")
            return
        
        self.target_marker = click_point
        self.stage_thread.request_move_to_image_pixel_stage_calibration(click_x, click_y)
        self.update_status(f"Moving stage to bring point ({click_x}, {click_y}) to reference position", "info")
        self.marker_timer.start(300)
    
    def image_wheel_event(self, event):
        """Handle mouse wheel events for Z-axis control"""
        delta = event.angleDelta().y()
        
        if delta > 0:
            z_delta = self.z_step
            self.update_status(f"Z-axis up {self.z_step} μm", "info")
        else:
            z_delta = -self.z_step
            self.update_status(f"Z-axis down {self.z_step} μm", "info")
        
        self.stage_thread.request_move_z_relative(z_delta)
        event.accept()
    
    # Status and UI update methods
    def update_status(self, message, msg_type="info"):
        """Update the status label with appropriate styling"""
        self.status_label.setText(message)
        
        if msg_type == "error":
            self.status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.status_label.setStyleSheet("background-color: #CCFFCC; color: #006600; padding: 5px;")
    
    def show_error(self, error_message):
        """Show error message"""
        self.update_status(f"ERROR: {error_message}", "error")
        logger.error(error_message)
    
    def clear_markers(self):
        """Clear visual markers after timer expires"""
        if self.marker_timer.isActive():
            self.marker_timer.stop()
        
        if not self.calibration_mode_image_pixel_stage_calibration:
            self.current_calibration_point_image_pixel_stage_calibration = None
            
        self.target_marker = None
    
    # Stage calibration methods
    def show_calibration_point_image_pixel_stage_calibration(self, index, point):
        """Show a calibration point"""
        self.calibration_mode_image_pixel_stage_calibration = True
        self.calibration_point_index_image_pixel_stage_calibration = index
        self.current_calibration_point_image_pixel_stage_calibration = point
    
    def start_calibration_image_pixel_stage_calibration(self):
        """Start the calibration process"""
        self.calibration_mode_image_pixel_stage_calibration = True
        self.reference_point_mode_image_pixel_stage_calibration = False
        self.calibration_point_index_image_pixel_stage_calibration = -1
        self.current_calibration_point_image_pixel_stage_calibration = None
        self.stage_thread.request_calibration_image_pixel_stage_calibration()
    
    def on_calibration_complete_image_pixel_stage_calibration(self, success):
        """Handle calibration completion"""
        self.calibration_mode_image_pixel_stage_calibration = False
        if success:
            self.set_ref_point_btn_image_pixel_stage_calibration.setEnabled(True)
    
    def start_set_reference_point_image_pixel_stage_calibration(self):
        """Start the process of setting a reference point"""
        if not self.stage_thread.is_calibrated_image_pixel_stage_calibration:
            self.update_status("Stage not calibrated. Please calibrate first.", "warning")
            return
            
        self.reference_point_mode_image_pixel_stage_calibration = True
        self.calibration_mode_image_pixel_stage_calibration = False
        self.update_status("Click on the image to set the reference point", "warning")
    
    # Video recording methods
    def update_recording_status(self, message, msg_type="info"):
        """Update recording status label"""
        self.recording_status_label.setText(message)
        
        if msg_type == "error":
            self.recording_status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.recording_status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.recording_status_label.setStyleSheet("background-color: #CCFFCC; color: #006600; padding: 5px;")
    
    def start_recording(self):
        """Start video recording"""
        if not self.is_recording and self.camera_mode:
            width, height = self.current_image_size
            success = self.video_recording_thread.start_recording(width, height, 30)
            
            if success:
                self.is_recording = True
                self.start_recording_btn.setEnabled(False)
                self.stop_recording_btn.setEnabled(True)
                self.update_recording_status("Recording...", "warning")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording:
            success = self.video_recording_thread.stop_recording()
            
            if success:
                self.is_recording = False
                self.start_recording_btn.setEnabled(True)
                self.stop_recording_btn.setEnabled(False)
                self.update_recording_status("Recording stopped", "info")
    
    def on_recording_started(self):
        """Handle recording started signal"""
        self.update_recording_status("Recording in progress...", "warning")
    
    def on_recording_stopped(self, filename):
        """Handle recording stopped signal"""
        if filename:
            self.update_recording_status(f"Video saved: {os.path.basename(filename)}", "info")
        else:
            self.update_recording_status("Recording stopped", "info")
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Stop threads
        self.camera_thread.stop()
        self.stage_thread.stop()
        self.video_recording_thread.stop()
        self.concentric_circle_thread.stop()
        
        # Wait for threads to finish
        self.camera_thread.wait()
        self.stage_thread.wait()
        self.video_recording_thread.wait()
        self.concentric_circle_thread.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()