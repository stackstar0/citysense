"""
Pattern Recognition - Advanced pattern detection in urban systems (Dependency-Free)
Identifies trends, anomalies, and behavioral patterns in city data without external ML libraries
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import statistics
import math
import random
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    DAILY_CYCLE = "daily_cycle"
    WEEKLY_CYCLE = "weekly_cycle"
    SEASONAL = "seasonal"
    ANOMALY = "anomaly"
    TREND = "trend"
    EVENT_DRIVEN = "event_driven"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"

@dataclass
class DetectedPattern:
    """Represents a detected pattern in urban data"""
    pattern_id: str
    pattern_type: PatternType
    confidence: float  # 0-1
    description: str
    affected_sensors: List[str]
    time_range: Tuple[datetime, datetime]
    parameters: Dict[str, Any]
    impact_score: float  # 0-1, how significant this pattern is
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class AnomalyResult:
    """Represents an anomaly detection result"""
    timestamp: datetime
    sensor_id: str
    value: float
    expected_value: float
    deviation_score: float
    anomaly_type: str
    severity: str
    context: Dict[str, Any]

class CustomKMeans:
    """Lightweight K-means clustering implementation"""
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = []
        self.labels = []
        self.is_fitted = False

    def fit(self, X: List[List[float]]):
        """Fit K-means clustering"""
        if not X or len(X) < self.n_clusters:
            self.is_fitted = True
            return

        n_features = len(X[0]) if X else 0
        n_samples = len(X)

        # Initialize centroids randomly
        self.centroids = []
        for _ in range(self.n_clusters):
            centroid = []
            for j in range(n_features):
                # Random initialization within data range
                feature_values = [row[j] for row in X]
                min_val, max_val = min(feature_values), max(feature_values)
                centroid.append(random.uniform(min_val, max_val))
            self.centroids.append(centroid)

        # Iterative clustering
        for iteration in range(self.max_iters):
            # Assign points to clusters
            new_labels = []
            for point in X:
                distances = []
                for centroid in self.centroids:
                    dist = sum((p - c) ** 2 for p, c in zip(point, centroid)) ** 0.5
                    distances.append(dist)
                new_labels.append(distances.index(min(distances)))

            # Check for convergence
            if iteration > 0 and new_labels == self.labels:
                break

            self.labels = new_labels

            # Update centroids
            for k in range(self.n_clusters):
                cluster_points = [X[i] for i in range(n_samples) if self.labels[i] == k]
                if cluster_points:
                    new_centroid = []
                    for j in range(n_features):
                        new_centroid.append(statistics.mean([point[j] for point in cluster_points]))
                    self.centroids[k] = new_centroid

        self.is_fitted = True

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster labels for new data"""
        if not self.is_fitted:
            return [0] * len(X)

        labels = []
        for point in X:
            distances = []
            for centroid in self.centroids:
                dist = sum((p - c) ** 2 for p, c in zip(point, centroid)) ** 0.5
                distances.append(dist)
            labels.append(distances.index(min(distances)))

        return labels

class CustomDBSCAN:
    """Lightweight DBSCAN clustering implementation"""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []
        self.is_fitted = False

    def fit(self, X: List[List[float]]):
        """Fit DBSCAN clustering"""
        if not X:
            self.is_fitted = True
            return

        n_samples = len(X)
        self.labels = [-1] * n_samples  # -1 means noise
        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:  # Already processed
                continue

            # Find neighbors
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                # Start new cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        self.is_fitted = True

    def _get_neighbors(self, X: List[List[float]], point_idx: int) -> List[int]:
        """Get neighbors within eps distance"""
        neighbors = []
        point = X[point_idx]

        for i, other_point in enumerate(X):
            if i == point_idx:
                continue

            # Calculate Euclidean distance
            dist = sum((p - o) ** 2 for p, o in zip(point, other_point)) ** 0.5
            if dist <= self.eps:
                neighbors.append(i)

        return neighbors

    def _expand_cluster(self, X: List[List[float]], point_idx: int, neighbors: List[int], cluster_id: int):
        """Expand cluster from seed point"""
        self.labels[point_idx] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == -1:  # Was noise
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == -1:  # Unprocessed
                self.labels[neighbor_idx] = cluster_id

                # Find neighbors of neighbor
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)

            i += 1

class CustomIsolationForest:
    """Lightweight isolation forest for anomaly detection"""
    def __init__(self, n_estimators=10, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.trees = []
        self.threshold = None
        self.is_fitted = False

    def fit(self, X: List[List[float]]):
        """Fit isolation forest"""
        if not X:
            self.is_fitted = True
            return

        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        self.trees = []

        # Build isolation trees
        for _ in range(self.n_estimators):
            # Sample subset of data
            sample_size = min(256, n_samples)
            sample_indices = random.sample(range(n_samples), sample_size)
            sample_data = [X[i] for i in sample_indices]

            # Build isolation tree
            tree = self._build_isolation_tree(sample_data, 0, math.ceil(math.log2(sample_size)))
            self.trees.append(tree)

        # Calculate threshold
        scores = self._calculate_scores(X)
        scores.sort()
        threshold_idx = int((1 - self.contamination) * len(scores))
        self.threshold = scores[threshold_idx] if threshold_idx < len(scores) else scores[-1]

        self.is_fitted = True

    def _build_isolation_tree(self, data: List[List[float]], depth: int, max_depth: int) -> Dict[str, Any]:
        """Build a single isolation tree"""
        if depth >= max_depth or len(data) <= 1:
            return {'type': 'leaf', 'size': len(data)}

        n_features = len(data[0]) if data else 0
        if n_features == 0:
            return {'type': 'leaf', 'size': len(data)}

        # Random split
        split_feature = random.randint(0, n_features - 1)
        feature_values = [row[split_feature] for row in data]
        min_val, max_val = min(feature_values), max(feature_values)

        if min_val == max_val:
            return {'type': 'leaf', 'size': len(data)}

        split_value = random.uniform(min_val, max_val)

        # Split data
        left_data = [row for row in data if row[split_feature] < split_value]
        right_data = [row for row in data if row[split_feature] >= split_value]

        return {
            'type': 'internal',
            'feature': split_feature,
            'threshold': split_value,
            'left': self._build_isolation_tree(left_data, depth + 1, max_depth),
            'right': self._build_isolation_tree(right_data, depth + 1, max_depth)
        }

    def _calculate_scores(self, X: List[List[float]]) -> List[float]:
        """Calculate anomaly scores for data points"""
        scores = []

        for point in X:
            path_lengths = []

            for tree in self.trees:
                path_length = self._get_path_length(point, tree, 0)
                path_lengths.append(path_length)

            avg_path_length = statistics.mean(path_lengths)
            # Normalize score (higher = more anomalous)
            score = 2 ** (-avg_path_length / self._c(len(X)))
            scores.append(score)

        return scores

    def _get_path_length(self, point: List[float], tree: Dict[str, Any], depth: int) -> float:
        """Get path length for a point in a tree"""
        if tree['type'] == 'leaf':
            return depth + self._c(tree['size'])

        if point[tree['feature']] < tree['threshold']:
            return self._get_path_length(point, tree['left'], depth + 1)
        else:
            return self._get_path_length(point, tree['right'], depth + 1)

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST"""
        if n <= 1:
            return 0
        return 2 * (math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        if not self.is_fitted:
            return [1] * len(X)

        scores = self._calculate_scores(X)
        return [-1 if score > self.threshold else 1 for score in scores]

class CustomPCA:
    """Lightweight Principal Component Analysis implementation"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = []
        self.mean = []
        self.is_fitted = False

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """Fit PCA and transform data"""
        if not X or not X[0]:
            return X

        n_samples = len(X)
        n_features = len(X[0])

        # Center the data
        self.mean = []
        for j in range(n_features):
            feature_values = [X[i][j] for i in range(n_samples)]
            self.mean.append(statistics.mean(feature_values))

        # Center data
        X_centered = []
        for i in range(n_samples):
            centered_row = [X[i][j] - self.mean[j] for j in range(n_features)]
            X_centered.append(centered_row)

        # Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X_centered)

        # Find eigenvalues and eigenvectors (simplified approach)
        eigenvalues, eigenvectors = self._power_iteration_eigendecomposition(cov_matrix, self.n_components)

        self.components = eigenvectors
        self.is_fitted = True

        # Transform data
        return self.transform(X)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Transform data using fitted PCA"""
        if not self.is_fitted:
            return X

        # Center data
        X_centered = []
        for row in X:
            centered_row = [row[j] - self.mean[j] for j in range(len(row))]
            X_centered.append(centered_row)

        # Project onto principal components
        transformed = []
        for row in X_centered:
            transformed_row = []
            for component in self.components:
                projection = sum(row[j] * component[j] for j in range(len(row)))
                transformed_row.append(projection)
            transformed.append(transformed_row)

        return transformed

    def _compute_covariance_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """Compute covariance matrix"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        cov_matrix = [[0.0] * n_features for _ in range(n_features)]

        for i in range(n_features):
            for j in range(n_features):
                covariance = sum(X[k][i] * X[k][j] for k in range(n_samples)) / (n_samples - 1)
                cov_matrix[i][j] = covariance

        return cov_matrix

    def _power_iteration_eigendecomposition(self, matrix: List[List[float]], n_components: int) -> Tuple[List[float], List[List[float]]]:
        """Simple eigendecomposition using power iteration"""
        n = len(matrix)
        eigenvalues = []
        eigenvectors = []

        # Make a copy of the matrix
        A = [row[:] for row in matrix]

        for _ in range(min(n_components, n)):
            # Initialize random vector
            v = [random.gauss(0, 1) for _ in range(n)]

            # Power iteration
            for _ in range(100):  # Fixed number of iterations
                # Matrix-vector multiplication
                Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]

                # Normalize
                norm = sum(x ** 2 for x in Av) ** 0.5
                if norm > 1e-10:
                    v = [x / norm for x in Av]
                else:
                    break

            # Calculate eigenvalue
            Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            eigenvalue = sum(v[i] * Av[i] for i in range(n))

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

            # Deflate matrix (remove this eigenvalue/eigenvector)
            for i in range(n):
                for j in range(n):
                    A[i][j] -= eigenvalue * v[i] * v[j]

        return eigenvalues, eigenvectors

class StandardScaler:
    """Lightweight feature scaler"""
    def __init__(self):
        self.mean = []
        self.std = []
        self.is_fitted = False

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """Fit and transform data"""
        if not X or not X[0]:
            return X

        n_features = len(X[0])
        self.mean = []
        self.std = []

        # Calculate mean and std for each feature
        for j in range(n_features):
            feature_values = [row[j] for row in X]
            mean_val = statistics.mean(feature_values)
            std_val = statistics.stdev(feature_values) if len(feature_values) > 1 else 1.0
            self.mean.append(mean_val)
            self.std.append(std_val if std_val > 0 else 1.0)

        self.is_fitted = True
        return self.transform(X)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            return X

        transformed = []
        for row in X:
            transformed_row = []
            for j, val in enumerate(row):
                scaled_val = (val - self.mean[j]) / self.std[j]
                transformed_row.append(scaled_val)
            transformed.append(transformed_row)

        return transformed

class UrbanPatternRecognition:
    """Advanced pattern recognition system for urban data"""

    def __init__(self):
        self.detected_patterns = []
        self.anomaly_history = deque(maxlen=1000)
        self.trend_data = defaultdict(deque)
        self.pattern_cache = {}

        # Initialize clustering and anomaly detection models
        self.kmeans = CustomKMeans(n_clusters=5)
        self.dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
        self.isolation_forest = CustomIsolationForest(n_estimators=20)
        self.pca = CustomPCA(n_components=3)
        self.scaler = StandardScaler()

        logger.info("Urban Pattern Recognition system initialized")

    async def analyze_patterns(self, sensor_data: Dict[str, Any],
                             historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Analyze patterns in urban sensor data"""
        try:
            patterns = []

            # Detect different types of patterns
            daily_patterns = await self._detect_daily_patterns(sensor_data, historical_data)
            patterns.extend(daily_patterns)

            weekly_patterns = await self._detect_weekly_patterns(sensor_data, historical_data)
            patterns.extend(weekly_patterns)

            anomaly_patterns = await self._detect_anomalies(sensor_data, historical_data)
            patterns.extend(anomaly_patterns)

            trend_patterns = await self._detect_trends(sensor_data, historical_data)
            patterns.extend(trend_patterns)

            correlation_patterns = await self._detect_correlations(sensor_data)
            patterns.extend(correlation_patterns)

            clustering_patterns = await self._detect_clustering_patterns(sensor_data, historical_data)
            patterns.extend(clustering_patterns)

            # Store detected patterns
            self.detected_patterns.extend(patterns)

            # Keep only recent patterns (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.detected_patterns = [
                p for p in self.detected_patterns
                if p.time_range[1] > cutoff_time
            ]

            logger.info(f"Detected {len(patterns)} patterns in current analysis")
            return patterns

        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return []

    async def _detect_daily_patterns(self, sensor_data: Dict[str, Any],
                                   historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Detect daily cyclical patterns"""
        patterns = []

        try:
            current_hour = datetime.now().hour

            # Analyze each sensor type for daily patterns
            iot_sensors = sensor_data.get('iot_sensors', {})

            for sensor_id, sensor_info in iot_sensors.items():
                try:
                    current_value = float(sensor_info.get('value', 50.0))
                    sensor_type = sensor_info.get('type', 'unknown')

                    # Generate expected daily pattern
                    expected_pattern = self._generate_daily_pattern(sensor_type, current_hour)
                    deviation = abs(current_value - expected_pattern) / max(expected_pattern, 1.0)

                    if deviation < 0.3:  # Within expected daily pattern
                        confidence = 0.8 - deviation

                        pattern = DetectedPattern(
                            pattern_id=f"daily_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H')}",
                            pattern_type=PatternType.DAILY_CYCLE,
                            confidence=confidence,
                            description=f"Daily cycle pattern detected in {sensor_type} sensor",
                            affected_sensors=[sensor_id],
                            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                            parameters={
                                'cycle_type': 'daily',
                                'hour': current_hour,
                                'expected_value': expected_pattern,
                                'actual_value': current_value,
                                'deviation': deviation
                            },
                            impact_score=0.6,
                            recommendations=[
                                f"Monitor {sensor_type} sensor for continued daily pattern compliance",
                                "Adjust system parameters to match daily usage patterns"
                            ],
                            metadata={
                                'sensor_type': sensor_type,
                                'pattern_strength': confidence,
                                'cycle_phase': self._get_cycle_phase(current_hour)
                            }
                        )

                        patterns.append(pattern)

                except Exception as e:
                    logger.warning(f"Error analyzing daily pattern for sensor {sensor_id}: {str(e)}")

            return patterns

        except Exception as e:
            logger.error(f"Error in daily pattern detection: {str(e)}")
            return []

    async def _detect_weekly_patterns(self, sensor_data: Dict[str, Any],
                                    historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Detect weekly cyclical patterns"""
        patterns = []

        try:
            current_day = datetime.now().weekday()  # 0 = Monday, 6 = Sunday
            current_hour = datetime.now().hour

            iot_sensors = sensor_data.get('iot_sensors', {})

            for sensor_id, sensor_info in iot_sensors.items():
                try:
                    current_value = float(sensor_info.get('value', 50.0))
                    sensor_type = sensor_info.get('type', 'unknown')

                    # Generate expected weekly pattern
                    expected_weekly = self._generate_weekly_pattern(sensor_type, current_day, current_hour)
                    weekly_deviation = abs(current_value - expected_weekly) / max(expected_weekly, 1.0)

                    if weekly_deviation < 0.25:  # Weekly pattern detected
                        confidence = 0.7 - weekly_deviation

                        pattern = DetectedPattern(
                            pattern_id=f"weekly_{sensor_id}_{datetime.now().strftime('%Y%m%d')}",
                            pattern_type=PatternType.WEEKLY_CYCLE,
                            confidence=confidence,
                            description=f"Weekly cycle pattern detected in {sensor_type} sensor",
                            affected_sensors=[sensor_id],
                            time_range=(datetime.now() - timedelta(hours=2), datetime.now()),
                            parameters={
                                'cycle_type': 'weekly',
                                'day_of_week': current_day,
                                'hour': current_hour,
                                'expected_value': expected_weekly,
                                'actual_value': current_value,
                                'deviation': weekly_deviation
                            },
                            impact_score=0.7,
                            recommendations=[
                                f"Optimize {sensor_type} system for weekly usage patterns",
                                "Consider weekend vs weekday operational adjustments"
                            ],
                            metadata={
                                'is_weekend': current_day >= 5,
                                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_day],
                                'pattern_strength': confidence
                            }
                        )

                        patterns.append(pattern)

                except Exception as e:
                    logger.warning(f"Error analyzing weekly pattern for sensor {sensor_id}: {str(e)}")

            return patterns

        except Exception as e:
            logger.error(f"Error in weekly pattern detection: {str(e)}")
            return []

    async def _detect_anomalies(self, sensor_data: Dict[str, Any],
                              historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Detect anomalous patterns in sensor data"""
        patterns = []

        try:
            iot_sensors = sensor_data.get('iot_sensors', {})

            # Prepare data for anomaly detection
            sensor_values = []
            sensor_ids = []

            for sensor_id, sensor_info in iot_sensors.items():
                try:
                    value = float(sensor_info.get('value', 50.0))
                    sensor_values.append([value])
                    sensor_ids.append(sensor_id)
                except:
                    continue

            if len(sensor_values) < 3:  # Need minimum data for anomaly detection
                return patterns

            # Scale the data
            scaled_values = self.scaler.fit_transform(sensor_values)

            # Detect anomalies using isolation forest
            self.isolation_forest.fit(scaled_values)
            anomaly_predictions = self.isolation_forest.predict(scaled_values)

            # Process anomaly results
            for i, (sensor_id, prediction) in enumerate(zip(sensor_ids, anomaly_predictions)):
                if prediction == -1:  # Anomaly detected
                    try:
                        sensor_info = iot_sensors[sensor_id]
                        current_value = float(sensor_info.get('value', 50.0))
                        sensor_type = sensor_info.get('type', 'unknown')

                        # Calculate anomaly severity
                        expected_value = self._calculate_expected_value(sensor_type, datetime.now())
                        deviation = abs(current_value - expected_value) / max(expected_value, 1.0)

                        severity = "High" if deviation > 0.5 else "Medium" if deviation > 0.2 else "Low"
                        confidence = min(0.9, 0.5 + deviation)

                        pattern = DetectedPattern(
                            pattern_id=f"anomaly_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                            pattern_type=PatternType.ANOMALY,
                            confidence=confidence,
                            description=f"Anomaly detected in {sensor_type} sensor - {severity} severity",
                            affected_sensors=[sensor_id],
                            time_range=(datetime.now() - timedelta(minutes=30), datetime.now()),
                            parameters={
                                'anomaly_type': 'statistical_outlier',
                                'current_value': current_value,
                                'expected_value': expected_value,
                                'deviation': deviation,
                                'severity': severity
                            },
                            impact_score=0.3 + deviation * 0.7,
                            recommendations=[
                                f"Investigate {sensor_type} sensor for potential issues",
                                "Check for equipment malfunction or environmental factors",
                                "Consider maintenance or calibration if pattern persists"
                            ],
                            metadata={
                                'detection_method': 'isolation_forest',
                                'sensor_type': sensor_type,
                                'timestamp': datetime.now().isoformat()
                            }
                        )

                        patterns.append(pattern)

                        # Store in anomaly history
                        anomaly_result = AnomalyResult(
                            timestamp=datetime.now(),
                            sensor_id=sensor_id,
                            value=current_value,
                            expected_value=expected_value,
                            deviation_score=deviation,
                            anomaly_type='statistical_outlier',
                            severity=severity,
                            context={'sensor_type': sensor_type}
                        )
                        self.anomaly_history.append(anomaly_result)

                    except Exception as e:
                        logger.warning(f"Error processing anomaly for sensor {sensor_id}: {str(e)}")

            return patterns

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return []

    async def _detect_trends(self, sensor_data: Dict[str, Any],
                           historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Detect trending patterns in sensor data"""
        patterns = []

        try:
            iot_sensors = sensor_data.get('iot_sensors', {})

            for sensor_id, sensor_info in iot_sensors.items():
                try:
                    current_value = float(sensor_info.get('value', 50.0))
                    sensor_type = sensor_info.get('type', 'unknown')

                    # Add to trend data history
                    self.trend_data[sensor_id].append({
                        'timestamp': datetime.now(),
                        'value': current_value
                    })

                    # Keep only last 50 readings for trend analysis
                    if len(self.trend_data[sensor_id]) > 50:
                        self.trend_data[sensor_id].popleft()

                    # Analyze trend if we have enough data
                    if len(self.trend_data[sensor_id]) >= 10:
                        trend_info = self._analyze_trend(self.trend_data[sensor_id])

                        if trend_info['significant']:
                            direction = "Increasing" if trend_info['slope'] > 0 else "Decreasing"
                            confidence = min(0.9, abs(trend_info['slope']) * 10)

                            pattern = DetectedPattern(
                                pattern_id=f"trend_{sensor_id}_{datetime.now().strftime('%Y%m%d_%H')}",
                                pattern_type=PatternType.TREND,
                                confidence=confidence,
                                description=f"{direction} trend detected in {sensor_type} sensor",
                                affected_sensors=[sensor_id],
                                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                                parameters={
                                    'trend_direction': direction.lower(),
                                    'slope': trend_info['slope'],
                                    'r_squared': trend_info['r_squared'],
                                    'data_points': len(self.trend_data[sensor_id])
                                },
                                impact_score=confidence,
                                recommendations=[
                                    f"Monitor {sensor_type} sensor for continued {direction.lower()} trend",
                                    "Investigate potential causes for trend pattern",
                                    "Consider adjusting system parameters if trend is undesirable"
                                ],
                                metadata={
                                    'sensor_type': sensor_type,
                                    'trend_strength': confidence,
                                    'analysis_window': '1_hour'
                                }
                            )

                            patterns.append(pattern)

                except Exception as e:
                    logger.warning(f"Error analyzing trend for sensor {sensor_id}: {str(e)}")

            return patterns

        except Exception as e:
            logger.error(f"Error in trend detection: {str(e)}")
            return []

    async def _detect_correlations(self, sensor_data: Dict[str, Any]) -> List[DetectedPattern]:
        """Detect correlation patterns between sensors"""
        patterns = []

        try:
            iot_sensors = sensor_data.get('iot_sensors', {})

            if len(iot_sensors) < 2:
                return patterns

            # Prepare sensor data for correlation analysis
            sensor_pairs = []
            sensor_values = []
            sensor_ids = list(iot_sensors.keys())

            for sensor_id in sensor_ids:
                try:
                    value = float(iot_sensors[sensor_id].get('value', 50.0))
                    sensor_values.append(value)
                except:
                    sensor_values.append(50.0)

            # Calculate correlations between sensor pairs
            for i in range(len(sensor_ids)):
                for j in range(i + 1, len(sensor_ids)):
                    sensor1_id = sensor_ids[i]
                    sensor2_id = sensor_ids[j]

                    # Use historical trend data if available for better correlation
                    if (sensor1_id in self.trend_data and sensor2_id in self.trend_data and
                        len(self.trend_data[sensor1_id]) >= 5 and len(self.trend_data[sensor2_id]) >= 5):

                        values1 = [point['value'] for point in list(self.trend_data[sensor1_id])[-10:]]
                        values2 = [point['value'] for point in list(self.trend_data[sensor2_id])[-10:]]

                        correlation = self._calculate_correlation(values1, values2)

                        if abs(correlation) > 0.7:  # Strong correlation
                            correlation_type = "Positive" if correlation > 0 else "Negative"
                            confidence = abs(correlation)

                            sensor1_type = iot_sensors[sensor1_id].get('type', 'unknown')
                            sensor2_type = iot_sensors[sensor2_id].get('type', 'unknown')

                            pattern = DetectedPattern(
                                pattern_id=f"correlation_{sensor1_id}_{sensor2_id}_{datetime.now().strftime('%Y%m%d_%H')}",
                                pattern_type=PatternType.CORRELATION,
                                confidence=confidence,
                                description=f"{correlation_type} correlation between {sensor1_type} and {sensor2_type} sensors",
                                affected_sensors=[sensor1_id, sensor2_id],
                                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                                parameters={
                                    'correlation_coefficient': correlation,
                                    'correlation_type': correlation_type.lower(),
                                    'sensor1_type': sensor1_type,
                                    'sensor2_type': sensor2_type,
                                    'data_points': min(len(values1), len(values2))
                                },
                                impact_score=confidence * 0.8,
                                recommendations=[
                                    f"Leverage {correlation_type.lower()} correlation for predictive maintenance",
                                    "Consider integrated monitoring for correlated sensors",
                                    "Use correlation for cross-validation of sensor readings"
                                ],
                                metadata={
                                    'analysis_method': 'pearson_correlation',
                                    'significance': 'high' if abs(correlation) > 0.8 else 'medium'
                                }
                            )

                            patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Error in correlation detection: {str(e)}")
            return []

    async def _detect_clustering_patterns(self, sensor_data: Dict[str, Any],
                                        historical_data: List[Dict[str, Any]] = None) -> List[DetectedPattern]:
        """Detect clustering patterns in sensor data"""
        patterns = []

        try:
            iot_sensors = sensor_data.get('iot_sensors', {})

            if len(iot_sensors) < 3:
                return patterns

            # Prepare data for clustering
            sensor_features = []
            sensor_ids = []

            for sensor_id, sensor_info in iot_sensors.items():
                try:
                    value = float(sensor_info.get('value', 50.0))
                    location = sensor_info.get('location', 'unknown')
                    sensor_type = sensor_info.get('type', 'unknown')

                    # Create feature vector (value, type_encoded, location_encoded)
                    type_encoded = hash(sensor_type) % 100
                    location_encoded = hash(location) % 100

                    sensor_features.append([value, type_encoded, location_encoded])
                    sensor_ids.append(sensor_id)

                except:
                    continue

            if len(sensor_features) < 3:
                return patterns

            # Perform clustering
            scaled_features = self.scaler.fit_transform(sensor_features)
            self.kmeans.fit(scaled_features)
            cluster_labels = self.kmeans.predict(scaled_features)

            # Analyze clusters
            cluster_groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                cluster_groups[label].append({
                    'sensor_id': sensor_ids[i],
                    'features': sensor_features[i],
                    'sensor_info': iot_sensors[sensor_ids[i]]
                })

            # Create patterns for significant clusters
            for cluster_id, sensors_in_cluster in cluster_groups.items():
                if len(sensors_in_cluster) >= 2:  # Minimum cluster size

                    cluster_sensor_ids = [s['sensor_id'] for s in sensors_in_cluster]
                    cluster_values = [s['features'][0] for s in sensors_in_cluster]
                    cluster_types = [s['sensor_info'].get('type', 'unknown') for s in sensors_in_cluster]

                    # Calculate cluster characteristics
                    avg_value = statistics.mean(cluster_values)
                    value_std = statistics.stdev(cluster_values) if len(cluster_values) > 1 else 0

                    # Determine cluster coherence
                    coherence = 1.0 - (value_std / max(avg_value, 1.0))
                    coherence = max(0.1, min(1.0, coherence))

                    if coherence > 0.6:  # Coherent cluster
                        pattern = DetectedPattern(
                            pattern_id=f"cluster_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H')}",
                            pattern_type=PatternType.CLUSTERING,
                            confidence=coherence,
                            description=f"Sensor cluster pattern detected - {len(cluster_sensor_ids)} sensors",
                            affected_sensors=cluster_sensor_ids,
                            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                            parameters={
                                'cluster_id': cluster_id,
                                'cluster_size': len(sensors_in_cluster),
                                'average_value': avg_value,
                                'value_std': value_std,
                                'coherence': coherence,
                                'sensor_types': list(set(cluster_types))
                            },
                            impact_score=coherence * 0.7,
                            recommendations=[
                                "Monitor cluster for coordinated behavior patterns",
                                "Consider cluster-based optimization strategies",
                                "Use cluster information for fault isolation"
                            ],
                            metadata={
                                'clustering_method': 'kmeans',
                                'cluster_coherence': coherence,
                                'dominant_types': cluster_types
                            }
                        )

                        patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Error in clustering pattern detection: {str(e)}")
            return []

    def _generate_daily_pattern(self, sensor_type: str, hour: int) -> float:
        """Generate expected daily pattern for sensor type"""
        patterns = {
            'air_quality': lambda h: 80 + 20 * math.sin((h - 6) * math.pi / 12),
            'temperature': lambda h: 22 + 8 * math.sin((h - 6) * math.pi / 12),
            'humidity': lambda h: 45 + 15 * math.cos(h * math.pi / 12),
            'co2': lambda h: 400 + 100 * math.sin((h - 3) * math.pi / 12),
            'noise': lambda h: 50 + 20 * (1 if 6 <= h <= 22 else 0.3),
            'motion': lambda h: 30 + 40 * (1 if 6 <= h <= 22 else 0.1)
        }

        pattern_func = patterns.get(sensor_type, lambda h: 50 + 10 * math.sin(h * math.pi / 12))
        return pattern_func(hour)

    def _generate_weekly_pattern(self, sensor_type: str, day: int, hour: int) -> float:
        """Generate expected weekly pattern for sensor type"""
        daily_base = self._generate_daily_pattern(sensor_type, hour)

        # Weekly factors (Monday=0 to Sunday=6)
        weekly_factors = {
            'air_quality': [1.0, 1.1, 1.1, 1.0, 0.9, 0.7, 0.8],
            'temperature': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Less weekly variation
            'humidity': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'co2': [1.0, 1.2, 1.2, 1.1, 1.0, 0.6, 0.7],
            'noise': [0.8, 1.2, 1.2, 1.1, 1.3, 1.0, 0.9],
            'motion': [0.7, 1.3, 1.3, 1.2, 1.4, 1.1, 0.8]
        }

        factor = weekly_factors.get(sensor_type, [1.0] * 7)[day]
        return daily_base * factor

    def _calculate_expected_value(self, sensor_type: str, timestamp: datetime) -> float:
        """Calculate expected value for sensor at given time"""
        hour = timestamp.hour
        day = timestamp.weekday()

        # Combine daily and weekly patterns
        daily_value = self._generate_daily_pattern(sensor_type, hour)
        weekly_factor = self._generate_weekly_pattern(sensor_type, day, hour) / daily_value

        return daily_value * weekly_factor

    def _get_cycle_phase(self, hour: int) -> str:
        """Get cycle phase for given hour"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _analyze_trend(self, data_points: deque) -> Dict[str, Any]:
        """Analyze trend in time series data"""
        if len(data_points) < 3:
            return {'significant': False, 'slope': 0, 'r_squared': 0}

        # Extract values and time indices
        values = [point['value'] for point in data_points]
        times = list(range(len(values)))

        # Calculate linear regression
        n = len(values)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(times, values))
        sum_x2 = sum(x * x for x in times)

        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return {'significant': False, 'slope': 0, 'r_squared': 0}

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(times, values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine if trend is significant
        significant = abs(slope) > 0.1 and r_squared > 0.3

        return {
            'significant': significant,
            'slope': slope,
            'r_squared': r_squared,
            'intercept': intercept
        }

    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0

        n = len(values1)

        # Calculate means
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n

        # Calculate correlation coefficient
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        sum_sq1 = sum((x - mean1) ** 2 for x in values1)
        sum_sq2 = sum((y - mean2) ** 2 for y in values2)

        denominator = (sum_sq1 * sum_sq2) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    async def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns"""
        try:
            summary = {
                'total_patterns': len(self.detected_patterns),
                'patterns_by_type': defaultdict(int),
                'recent_anomalies': len([a for a in self.anomaly_history if a.timestamp > datetime.now() - timedelta(hours=1)]),
                'high_confidence_patterns': 0,
                'sensors_with_patterns': set()
            }

            for pattern in self.detected_patterns:
                summary['patterns_by_type'][pattern.pattern_type.value] += 1
                if pattern.confidence > 0.7:
                    summary['high_confidence_patterns'] += 1
                summary['sensors_with_patterns'].update(pattern.affected_sensors)

            summary['sensors_with_patterns'] = len(summary['sensors_with_patterns'])
            summary['patterns_by_type'] = dict(summary['patterns_by_type'])

            return summary

        except Exception as e:
            logger.error(f"Error generating pattern summary: {str(e)}")
            return {}

    async def get_anomaly_report(self) -> List[Dict[str, Any]]:
        """Get detailed anomaly report"""
        try:
            recent_anomalies = [
                {
                    'timestamp': anomaly.timestamp.isoformat(),
                    'sensor_id': anomaly.sensor_id,
                    'value': anomaly.value,
                    'expected_value': anomaly.expected_value,
                    'deviation_score': anomaly.deviation_score,
                    'severity': anomaly.severity,
                    'sensor_type': anomaly.context.get('sensor_type', 'unknown')
                }
                for anomaly in list(self.anomaly_history)[-20:]  # Last 20 anomalies
            ]

            return recent_anomalies

        except Exception as e:
            logger.error(f"Error generating anomaly report: {str(e)}")
            return []

# Global instance
pattern_recognition = UrbanPatternRecognition()

async def main():
    """Test the pattern recognition system"""
    try:
        logger.info("Testing Urban Pattern Recognition...")

        # Test data
        test_data = {
            'iot_sensors': {
                'air_quality_001': {
                    'type': 'air_quality',
                    'value': 85.2,
                    'location': 'downtown',
                    'status': 'active'
                },
                'temperature_001': {
                    'type': 'temperature',
                    'value': 24.5,
                    'location': 'downtown',
                    'status': 'active'
                },
                'humidity_001': {
                    'type': 'humidity',
                    'value': 42.3,
                    'location': 'downtown',
                    'status': 'active'
                },
                'co2_001': {
                    'type': 'co2',
                    'value': 450.0,
                    'location': 'residential',
                    'status': 'active'
                }
            }
        }

        # Test pattern analysis
        patterns = await pattern_recognition.analyze_patterns(test_data)
        logger.info(f"Detected {len(patterns)} patterns")

        # Test pattern summary
        summary = await pattern_recognition.get_pattern_summary()
        logger.info(f"Pattern summary: {summary}")

        # Test anomaly report
        anomalies = await pattern_recognition.get_anomaly_report()
        logger.info(f"Recent anomalies: {len(anomalies)}")

        logger.info("Urban Pattern Recognition test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())