import os
import time
import queue
import logging
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QObject, pyqtSlot

class VideoRecorder(QObject):
    """Video recorder worker object that runs in a separate thread"""
    start_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self, filename, fps, frame_size):
        super().__init__()
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.start_signal.connect(self.start_recording)
        self.stop_signal.connect(self.stop_recording)
        self.isColor = False  # Grayscale by default
        self.logger = logging.getLogger(__name__)

    def start_recording(self):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename, 
            fourcc, 
            self.fps, 
            self.frame_size,
            self.isColor
        )
        if not self.writer.isOpened():
            self.logger.error(f"Failed to open video file {self.filename}")

    @pyqtSlot(np.ndarray)
    def write_frame(self, frame):
        """Write a frame to the video file"""
        if self.writer and self.writer.isOpened():
            # Ensure frame is the correct size
            if (frame.shape[1], frame.shape[0]) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            # Convert to grayscale if needed
            if len(frame.shape) == 3 and not self.isColor:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.writer.write(frame)

    def stop_recording(self):
        """Stop video recording and release resources"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info(f"Video saved: {self.filename}")


class VideoRecordingThread(QThread):
    """Thread for handling video recording"""
    status_signal = pyqtSignal(str, str)  # message, type
    recording_started_signal = pyqtSignal()
    recording_stopped_signal = pyqtSignal(str)  # filename
    
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.is_recording = False
        self.video_recorder = None
        self.video_thread = None
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Create savedImg directory if it doesn't exist
        os.makedirs("savedImg", exist_ok=True)
    
    def run(self):
        """Main thread loop"""
        self.running = True
        while self.running:
            if self.is_recording:
                try:
                    # Get frame from queue with timeout
                    frame = self.frame_queue.get(timeout=0.1)
                    if self.video_recorder:
                        # Emit frame to video recorder
                        self.video_recorder.write_frame(frame)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error writing frame: {e}")
            else:
                time.sleep(0.1)  # Sleep when not recording
    
    def update_frame(self, image):
        """Update the current frame for recording"""
        if self.is_recording:
            try:
                # Add frame to queue, drop oldest if full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(image.copy())
            except Exception as e:
                self.logger.error(f"Error updating frame: {e}")
    
    def start_recording(self, width=1600, height=1200, fps=30):
        """Start video recording"""
        self.mutex.lock()
        try:
            if self.is_recording:
                self.status_signal.emit("Already recording", "warning")
                return False
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"savedImg/capture_{timestamp}.mp4"
            
            # Create video recorder
            self.video_recorder = VideoRecorder(filename, fps, (width, height))
            
            # Start recording
            self.video_recorder.start_recording()
            self.is_recording = True
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status_signal.emit(f"Recording started: {filename}", "info")
            self.recording_started_signal.emit()
            self.logger.info(f"Started recording: {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.status_signal.emit(f"Failed to start recording: {str(e)}", "error")
            return False
        finally:
            self.mutex.unlock()
    
    def stop_recording(self):
        """Stop video recording"""
        self.mutex.lock()
        try:
            if not self.is_recording:
                self.status_signal.emit("Not currently recording", "warning")
                return False
            
            # Stop recording
            if self.video_recorder:
                filename = self.video_recorder.filename
                self.video_recorder.stop_recording()
                self.video_recorder = None
            
            self.is_recording = False
            
            # Clear remaining frames in queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status_signal.emit(f"Recording stopped and saved", "info")
            self.recording_stopped_signal.emit(filename if 'filename' in locals() else "")
            self.logger.info("Stopped recording")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            self.status_signal.emit(f"Failed to stop recording: {str(e)}", "error")
            return False
        finally:
            self.mutex.unlock()
    
    def stop(self):
        """Stop the thread"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        self.running = False
        self.wait(1000)