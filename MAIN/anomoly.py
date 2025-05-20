import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import pandas as pd
from typing import Dict, List, Tuple, Optional


class AnomalyDetector:
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.movement_history = deque(maxlen=window_size)
        self.is_trained = False

    def _prepare_features(self, finger_pos: Tuple[int, int],
                          muscle_pos: Tuple[int, int],
                          previous_finger_pos: Optional[Tuple[int, int]] = None) -> List[float]:
        features = list(finger_pos) + list(muscle_pos)

        if previous_finger_pos:
            # Calculate velocity
            velocity_x = finger_pos[0] - previous_finger_pos[0]
            velocity_y = finger_pos[1] - previous_finger_pos[1]
            features.extend([velocity_x, velocity_y])
        else:
            features.extend([0, 0])  # Default velocity when no previous position

        return features

    def add_movement(self, finger_pos: Tuple[int, int], muscle_pos: Tuple[int, int],
                     muscle_name: str) -> None:
        previous_finger_pos = self.movement_history[-1][0] if self.movement_history else None
        features = self._prepare_features(finger_pos, muscle_pos, previous_finger_pos)
        self.movement_history.append((finger_pos, muscle_pos, muscle_name, features))

    def train(self) -> None:
        if len(self.movement_history) < self.window_size // 2:
            return

        X = np.array([m[3] for m in self.movement_history])
        self.isolation_forest.fit(X)
        self.is_trained = True

    def detect_anomaly(self, finger_pos: Tuple[int, int],
                       muscle_pos: Tuple[int, int]) -> Tuple[bool, float]:
        if not self.is_trained or len(self.movement_history) < 2:
            return False, 0.0

        previous_finger_pos = self.movement_history[-1][0]
        features = self._prepare_features(finger_pos, muscle_pos, previous_finger_pos)

        # Reshape for single sample prediction
        features_reshaped = np.array(features).reshape(1, -1)

        # Get anomaly score
        anomaly_score = self.isolation_forest.score_samples(features_reshaped)[0]

        # Determine if it's an anomaly
        is_anomaly = self.isolation_forest.predict(features_reshaped)[0] == -1

        return is_anomaly, anomaly_score

    def get_movement_statistics(self) -> Dict:
        if not self.movement_history:
            return {}

        df = pd.DataFrame([
            {
                'finger_x': m[0][0],
                'finger_y': m[0][1],
                'muscle_x': m[1][0],
                'muscle_y': m[1][1],
                'muscle_name': m[2]
            }
            for m in self.movement_history
        ])

        stats = {
            'total_movements': len(self.movement_history),
            'unique_muscles': df['muscle_name'].nunique(),
            'most_common_muscle': df['muscle_name'].mode().iloc[0],
            'avg_finger_x': df['finger_x'].mean(),
            'avg_finger_y': df['finger_y'].mean()
        }

        return stats


def create_detector(window_size: int = 100) -> AnomalyDetector:
    return AnomalyDetector(window_size=window_size)