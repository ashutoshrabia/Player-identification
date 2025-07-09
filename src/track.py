# src/track.py
import numpy as np
from enum import Enum, auto
from collections import deque

class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED   = auto()

class Track:
    """
    Single player track with an 8-D constant-velocity Kalman filter,
    plus OCR-based jersey stabilization.
    """

    _count = 0

    def __init__(self, bbox, feature, max_age=50, hit_thresh=3):
        self.id = Track._count
        Track._count += 1

        self.time_since_update = 0
        self.hits = 1
        self.hit_thresh = hit_thresh
        self.max_age = max_age
        self.state = TrackState.TENTATIVE
        self.feature = feature.copy()

        # Jersey OCR buffer
        self.jersey = None
        self.ocr_history = deque(maxlen=5)

        # Initialize Kalman filter
        self.dt = 1.0
        self._init_kalman(bbox)

    def _init_kalman(self, bbox):
       
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)

        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0

        # Transition matrix A
        self.A = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.A[i, i+4] = self.dt

        # Measurement matrix H
        self.H = np.zeros((4,8), dtype=np.float32)
        self.H[:, :4] = np.eye(4, dtype=np.float32)

        # Process & measurement noise
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0

    @staticmethod
    def _xyxy_to_cxcywh(box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_xyxy(state):
        cx, cy, w, h = state[:4]
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self):
    
        self.x = self.A.dot(self.x)
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
        self.time_since_update += 1
        return self._cxcywh_to_xyxy(self.x)

    def update(self, bbox, feature, alpha=0.9):
        
        z = np.array(self._xyxy_to_cxcywh(bbox), dtype=np.float32)

        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)

        self.x = self.x + K.dot(y)
        self.P = (np.eye(8) - K.dot(self.H)).dot(self.P)

        
        self.feature = alpha * self.feature + (1 - alpha) * feature
        self.feature /= np.linalg.norm(self.feature) + 1e-6

        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.hit_thresh:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELETED

    def to_xyxy(self):
        return self._cxcywh_to_xyxy(self.x)

    def add_ocr(self, jersey_str):
        if jersey_str is not None:
            self.ocr_history.append(jersey_str)
        if len(self.ocr_history) == self.ocr_history.maxlen:
            s = set(self.ocr_history)
            if len(s) == 1:
                self.jersey = s.pop()