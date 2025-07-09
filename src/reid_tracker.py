import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track
from feature import FeatureExtractor


def iou(boxA: np.ndarray, boxB: np.ndarray) -> np.ndarray:
    """
    Compute IoU between arrays of boxes:
      boxA shape: (T,4), boxB shape: (D,4)
    Returns IoU matrix of shape (T, D).
    """
    if boxA.size == 0 or boxB.size == 0:
        return np.zeros((boxA.shape[0], boxB.shape[0]), dtype=float)
    xA = np.maximum(boxA[:, 0][:, None], boxB[:, 0])
    yA = np.maximum(boxA[:, 1][:, None], boxB[:, 1])
    xB = np.minimum(boxA[:, 2][:, None], boxB[:, 2])
    yB = np.minimum(boxA[:, 3][:, None], boxB[:, 3])

    interW = np.maximum(0, xB - xA)
    interH = np.maximum(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    areaB = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])
    union = areaA[:, None] + areaB - interArea + 1e-6

    return interArea / union


class PlayerTracker:
   
    def __init__(
        self,
        extract_fn,
        min_box_w: int = 50,
        min_box_h: int = 80,
        iou_w: float = 0.4,
        app_w: float = 0.4,
        jersey_w: float = 0.2,
        max_age: int = 50,
        hit_thresh: int = 3,
        feat_threshold: float = 0.65,
        gating_factor: float = 1.5,
        min_speed_thresh: float = 30.0,
        output_thresh: int = 5,
        dead_cooldown: int = 50  # frames to prevent ID reuse
    ):
        self.tracks = []
        self.dead_counter = {}  # track_id -> frames remaining before reuse allowed
        self.extractor = FeatureExtractor()
        self.extract_fn = extract_fn
        self.min_box_w = min_box_w
        self.min_box_h = min_box_h
        self.iou_w = iou_w
        self.app_w = app_w
        self.jersey_w = jersey_w
        self.max_age = max_age
        self.hit_thresh = hit_thresh
        self.feat_threshold = feat_threshold
        self.gating_factor = gating_factor
        self.min_speed_thresh = min_speed_thresh
        self.output_thresh = output_thresh
        self.dead_cooldown = dead_cooldown

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=float)
        return 1.0 - np.dot(a, b.T)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[dict]:
        
        for tid in list(self.dead_counter):
            self.dead_counter[tid] -= 1
            if self.dead_counter[tid] <= 0:
                del self.dead_counter[tid]

        
        if detections.shape[0] == 0:
            return self._prune_and_output()

        
        predicted_boxes = []
        speeds = []
        for trk in self.tracks:
            bbox = trk.predict()
            predicted_boxes.append(bbox)
            vx, vy = trk.x[4], trk.x[5]
            speeds.append(np.hypot(vx, vy))
        T = len(self.tracks)
        predicted_boxes = np.array(predicted_boxes) if T else np.empty((0,4))
        speeds = np.array(speeds) if T else np.array([])

    
        det_feats = np.array([
            self.extractor(frame, d) for d in detections
        ], dtype=np.float32)

        # Jersey OCR
        det_jerseys = []
        for bbox in detections:
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) >= self.min_box_w and (y2 - y1) >= self.min_box_h:
                crop = frame[y1:y2, x1:x2]
                det_jerseys.append(self.extract_fn(crop))
            else:
                det_jerseys.append(None)

        # Build cost and gating
        D = len(detections)
        cost_iou = 1.0 - iou(predicted_boxes, detections)
        cost_app = self._cosine_distance(
            np.array([t.feature for t in self.tracks]) if T else np.empty((0,)), det_feats
        )
        jersey_cost = np.ones((T, D), dtype=float)
        for i, trk in enumerate(self.tracks):
            for j, dj in enumerate(det_jerseys):
                if trk.jersey is not None and dj is not None:
                    jersey_cost[i,j] = 0.0 if trk.jersey == dj else 1.0

        cost = (
            self.iou_w * cost_iou
            + self.app_w * cost_app
            + self.jersey_w * jersey_cost
        )

        
        gating_mask = np.ones((T, D), dtype=bool)
        for i in range(T):
            speed = max(speeds[i] * self.gating_factor, self.min_speed_thresh)
            px1, py1, px2, py2 = predicted_boxes[i]
            pcx, pcy = (px1+px2)/2, (py1+py2)/2
            for j, bbox in enumerate(detections):
                x1, y1, x2, y2 = bbox
                dcx, dcy = (x1+x2)/2, (y1+y2)/2
                if np.hypot(dcx-pcx, dcy-pcy) > speed:
                    gating_mask[i,j] = False
        cost[~gating_mask] = 1e6

        
        for i in range(T):
            if self.tracks[i].id in self.dead_counter:
                cost[i,:] = 1e6

        
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, un_trk, un_det = [], [], []
        for r, c in zip(row_ind, col_ind):
            if r < T and c < D and cost_app[r,c] < self.feat_threshold:
                matches.append((r,c))
            else:
                if r < T: un_trk.append(r)
                if c < D: un_det.append(c)
        un_trk += [i for i in range(T) if i not in row_ind]
        un_det += [j for j in range(D) if j not in col_ind]

    
        for r,c in matches:
            trk = self.tracks[r]
            trk.update(detections[c], det_feats[c])
            trk.add_ocr(det_jerseys[c])

    
        for j in un_det:
            new_trk = Track(
                bbox=detections[j],
                feature=det_feats[j],
                max_age=self.max_age,
                hit_thresh=self.hit_thresh
            )
            new_trk.add_ocr(det_jerseys[j])
            self.tracks.append(new_trk)

        
        for trk in list(self.tracks):
            if trk.is_deleted():
                self.dead_counter[trk.id] = self.dead_cooldown

        
        return self._prune_and_output()

    def _prune_and_output(self) -> list[dict]:
        outputs = []
        for trk in list(self.tracks):
            trk.mark_missed()
            
            if not trk.is_confirmed() and trk.time_since_update > self.max_age:
                self.tracks.remove(trk)
            
            elif trk.is_deleted():
                self.tracks.remove(trk)
            
            elif trk.is_confirmed() and trk.hits >= self.output_thresh:
                outputs.append({"track_id":trk.id, "bbox":trk.to_xyxy()})
        return outputs
