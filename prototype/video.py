"""Honor video display orientation before preview and inference."""
import cv2

def configure_capture(cap):
    # OpenCV reports clockwise display rotation; ffprobe display matrices use
    # the opposite sign. Disable implicit rotation so it is applied exactly once.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    angle = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
    return angle if angle in (0, 90, 180, 270) else 0

def portrait_frame(frame, clockwise=0):
    if clockwise == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif clockwise == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif clockwise == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Untagged landscape camera frames also follow the requested portrait mode.
    if frame.shape[1] > frame.shape[0]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        clockwise = (clockwise + 90) % 360
    return frame, clockwise
