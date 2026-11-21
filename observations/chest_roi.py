from __future__ import division
try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime dependency fallback
    mp = None
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable or []
            self.format_dict = {'elapsed': 0.0}
        def __iter__(self):
            return iter(self.iterable)
        @staticmethod
        def write(msg):
            print(msg)
from PIL import Image
import cv2 as cv
import re 


_MP_SOLUTIONS_WARNED = False


def _get_mp_solutions():
    """Return mediapipe.solutions module when available, else None."""
    global _MP_SOLUTIONS_WARNED
    if mp is None:
        if not _MP_SOLUTIONS_WARNED:
            print("[WARN] mediapipe is not available. Using heuristic ROI fallback.")
            _MP_SOLUTIONS_WARNED = True
        return None
    solutions = getattr(mp, "solutions", None)
    if solutions is None and not _MP_SOLUTIONS_WARNED:
        print("[WARN] mediapipe.solutions is unavailable in this build. Using heuristic ROI fallback.")
        _MP_SOLUTIONS_WARNED = True
    return solutions


def _clip_bbox(xmin, xmax, ymin, ymax, width, height):
    xmin = max(int(round(xmin)), 0)
    xmax = min(int(round(xmax)), width)
    ymin = max(int(round(ymin)), 0)
    ymax = min(int(round(ymax)), height)
    if xmax <= xmin:
        xmax = min(width, xmin + 1)
    if ymax <= ymin:
        ymax = min(height, ymin + 1)
    return [xmin, xmax, ymin, ymax]


def _fallback_face_bbox(img):
    """Top-center face-like fallback when detection is unavailable."""
    image_height, image_width = img.shape[:2]
    side = int(0.35 * min(image_width, image_height))
    cx = image_width * 0.5
    cy = image_height * 0.28
    return _clip_bbox(cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2, image_width, image_height)


def _fallback_chest_bbox(img, face_bbox=None):
    """Deterministic chest ROI fallback derived from frame geometry (and face when available)."""
    image_height, image_width = img.shape[:2]
    if face_bbox is not None:
        fxmin, fxmax, fymin, fymax = face_bbox
        f_w = max(1.0, float(fxmax - fxmin))
        f_h = max(1.0, float(fymax - fymin))
        cx = (fxmin + fxmax) / 2.0
        cy = fymax + 0.9 * f_h
        chest_w = min(image_width * 0.85, 1.9 * f_w)
        chest_h = max(6.0, 0.23 * chest_w)
    else:
        cx = image_width * 0.5
        cy = image_height * 0.60
        chest_w = image_width * 0.40
        chest_h = max(6.0, image_height * 0.09)
    return _clip_bbox(cx - chest_w / 2, cx + chest_w / 2, cy - chest_h / 2, cy + chest_h / 2, image_width, image_height)



def get_vid_stats(videoFileName):
    cap = cv.VideoCapture(videoFileName)
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, int(fps)

def sort_nicely(l): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 
  return l

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def detect_face(img):
    image_height, image_width, _ = img.shape
    solutions = _get_mp_solutions()
    if solutions is None or not hasattr(solutions, "face_detection"):
        return _fallback_face_bbox(img)

    mp_face_detection = solutions.face_detection
    try:
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            detection_result = face_detection.process(img)

        if detection_result is None or not detection_result.detections:
            return _fallback_face_bbox(img)

        bbox = detection_result.detections[0].location_data.relative_bounding_box
        bbox_pxl = [
            bbox.xmin * image_width,
            bbox.ymin * image_height,
            bbox.width * image_width,
            bbox.height * image_height,
        ]
        xmin = bbox_pxl[0]
        xmax = xmin + bbox_pxl[2]
        ymin = bbox_pxl[1]
        ymax = ymin + bbox_pxl[3]
        centerx = xmax - (xmax - xmin) / 2
        centery = ymax - (ymax - ymin) / 2
        xdist = max(image_width - centerx, centerx)
        ydist = max(image_height - centery, centery)
        d = min(xdist, ydist)
        xmin = centerx - d
        xmax = centerx + d
        ymin = centery - d
        ymax = centery + d
        return _clip_bbox(xmin, xmax, ymin, ymax, image_width, image_height)
    except Exception:
        return _fallback_face_bbox(img)


def get_chest_ROI(video_path, dataset, mp_complexity=2, skip_rate=1):
    print("\nExtracting ROIs...")

    _, fps = get_vid_stats(video_path)
    update_every = max(1, int(skip_rate) if skip_rate is not None else 1)
    i = 0
    solutions = _get_mp_solutions()
    mp_pose = solutions.pose if (solutions is not None and hasattr(solutions, "pose")) else None

    frames = []

    # Run MediaPipe Pose when available; otherwise use deterministic heuristic fallback.
    t = tqdm(extract_frames_yield(video_path))
    if mp_pose is None:
        fallback_bbox = None
        for frame in t:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            if fallback_bbox is None:
                face_bbox = detect_face(frame)
                fallback_bbox = _fallback_chest_bbox(frame, face_bbox=face_bbox)
            im = Image.fromarray(frame)
            left, right, upper, lower = fallback_bbox[0], fallback_bbox[1], fallback_bbox[2], fallback_bbox[3]
            if upper >= lower:
                lower = upper + 1
            if left >= right:
                right = left + 1
            chest = im.crop(box=(left, upper, right, lower))
            frames.append(chest)
            i += 1
    else:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=mp_complexity) as pose:
            results = None
            patch_width = None
            patch_height = None
            fallback_bbox = None
            last_bbox = None
            for frame in t:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Keep ROI size fixed for OF stability, but refresh center periodically.
                if i == 0 or (i % update_every == 0):
                    results = pose.process(frame)

                image_height, image_width, _ = frame.shape

                # Get landmark.
                if results is not None and results.pose_landmarks is not None:
                    x_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
                    y_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
                    x_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
                    y_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

                    shoulder_span = max(float(abs(x_left - x_right)), image_width * 0.15)
                    if patch_width is None or patch_height is None:
                        patch_width = shoulder_span
                        patch_height = max(patch_width * 0.2, image_height * 0.05)

                    center_x = 0.5 * (x_left + x_right)
                    center_y = min(y_right, y_left)
                    left = center_x - patch_width / 2
                    right = center_x + patch_width / 2
                    upper = center_y - patch_height / 2
                    lower = center_y + patch_height / 2
                    left, right, upper, lower = _clip_bbox(left, right, upper, lower, image_width, image_height)
                    last_bbox = [left, right, upper, lower]
                else:
                    if last_bbox is not None:
                        left, right, upper, lower = last_bbox
                    else:
                        if fallback_bbox is None:
                            face_bbox = detect_face(frame)
                            fallback_bbox = _fallback_chest_bbox(frame, face_bbox=face_bbox)
                        left, right, upper, lower = fallback_bbox[0], fallback_bbox[1], fallback_bbox[2], fallback_bbox[3]
                        last_bbox = [left, right, upper, lower]

                im = Image.fromarray(frame)
                if upper >= lower:
                    lower = upper + 1
                if left >= right:
                    right = left + 1
                chest = im.crop(box=(left, upper, right, lower))
                frames.append(chest)
                i += 1
    elapsed = t.format_dict["elapsed"]
    return frames, fps, elapsed











def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)










