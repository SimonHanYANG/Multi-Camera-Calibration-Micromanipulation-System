import time
import logging
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

class CameraThread(QThread):
    new_image_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.frame_rate = 30.0
        self.logger = logging.getLogger(__name__)
        self.camera_available = False
        
    def check_camera_availability(self):
        """Check if camera is available"""
        try:
            import pypylon.pylon as pylon
            
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()
            
            # Find all available devices
            devices = tl_factory.EnumerateDevices()
            
            self.camera_available = len(devices) > 0
            return self.camera_available
            
        except ImportError:
            self.logger.error("Pypylon not installed")
            self.camera_available = False
            return False
        except Exception as e:
            self.logger.error(f"Error checking camera availability: {e}")
            self.camera_available = False
            return False
        
    def run(self):
        self.running = True
        
        if not self.check_camera_availability():
            self.error_signal.emit("No cameras found.")
            return
            
        try:
            # Import here to avoid import errors if not installed
            import pypylon.pylon as pylon
            from pypylon import genicam
            
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()
            
            # Find all available devices
            devices = tl_factory.EnumerateDevices()
            
            # Create and connect camera
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            logger.info(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # Open camera
            self.camera.Open()
            
            # Configure camera settings
            self._configure_camera()
            
            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            logger.info("Started grabbing images")
            
            while self.running and self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Copy array to avoid data race
                    img_array = grab_result.Array.copy()
                    self.new_image_signal.emit(img_array)
                
                grab_result.Release()
                
        except Exception as e:
            error_msg = f"Camera error: {str(e)}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        finally:
            self._cleanup()
    
    def _configure_camera(self):
        """Configure camera parameters"""
        try:
            import pypylon.pylon as pylon
            from pypylon import genicam
            
            # Set to continuous acquisition
            if genicam.IsAvailable(self.camera.TriggerMode):
                self.camera.TriggerMode.SetValue("Off")
            
            # Heartbeat timeout (for GigE cameras)
            if (self.camera.GetDeviceInfo().GetDeviceClass() == "BaslerGigE" and 
                genicam.IsAvailable(self.camera.GevHeartbeatTimeout)):
                self.camera.GevHeartbeatTimeout.SetValue(1000)
            
            # Frame rate settings
            if genicam.IsAvailable(self.camera.AcquisitionFrameRateEnable):
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                
                if genicam.IsAvailable(self.camera.AcquisitionFrameRateAbs):
                    self.camera.AcquisitionFrameRateAbs.SetValue(self.frame_rate)
                elif genicam.IsAvailable(self.camera.AcquisitionFrameRate):
                    self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
            
            # Pixel format
            if genicam.IsAvailable(self.camera.PixelFormat):
                self.camera.PixelFormat.SetValue("Mono8")
            
            # ROI settings (1600x1200)
            if (genicam.IsAvailable(self.camera.Width) and 
                genicam.IsAvailable(self.camera.Height)):
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
                self.camera.Width.SetValue(1600)
                self.camera.Height.SetValue(1200)
                
        except Exception as e:
            error_msg = f"Failed to configure camera: {str(e)}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
            raise
    
    def _cleanup(self):
        """Clean up camera resources"""
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                    logger.info("Stopped grabbing images")
                
                if self.camera.IsOpen():
                    logger.info(f"Closing camera {self.camera.GetDeviceInfo().GetModelName()}")
                    self.camera.Close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        self.running = False
        self.wait(3000)  # Wait up to 3 seconds for safe shutdown