"""
Pattern Recognition - Advanced pattern detection in urban systems (Dependency-Free)
Identifies trends, anomalies, and behavioral patterns in city data without sklearn/pandas
"""

import asyncio
import logging
import math
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

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

class PatternRecognition:
    """
    Advanced pattern recognition for urban intelligence systems.
    Detects patterns, anomalies, and trends without external ML dependencies.
    """

    def __init__(self):
        self.patterns = {}
        self.detection_history = []
        self.thresholds = {
            'anomaly_z_score': 2.5,
            'trend_min_points': 5,
            'correlation_threshold': 0.7,
            'pattern_confidence_min': 0.6
        }

    async def detect_patterns(self, sensor_data: Dict[str, List[Dict[str, Any]]]) -> List[DetectedPattern]:
        """Main pattern detection method"""
        patterns = []

        for sensor_id, data_points in sensor_data.items():
            if len(data_points) < 3:
                continue

            # Extract time series values
            values = [point.get('value', 0) for point in data_points]
            timestamps = [point.get('timestamp', datetime.now()) for point in data_points]

            # Detect different pattern types
            patterns.extend(await self._detect_anomalies(sensor_id, values, timestamps))
            patterns.extend(await self._detect_trends(sensor_id, values, timestamps))
            patterns.extend(await self._detect_cycles(sensor_id, values, timestamps))

        return patterns

    async def _detect_anomalies(self, sensor_id: str, values: List[float], timestamps: List[datetime]) -> List[DetectedPattern]:
        """Detect anomalous patterns using statistical methods"""
        patterns = []

        if len(values) < 5:
            return patterns

        # Calculate z-scores
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0

        if std_val == 0:
            return patterns

        z_scores = [(val - mean_val) / std_val for val in values]

        # Find anomalies
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > self.thresholds['anomaly_z_score']:
                pattern = DetectedPattern(
                    pattern_id=f"anomaly_{sensor_id}_{i}",
                    pattern_type=PatternType.ANOMALY,
                    confidence=min(0.95, abs(z_score) / 3.0),
                    description=f"Anomalous reading detected: {values[i]:.2f} (z-score: {z_score:.2f})",
                    affected_sensors=[sensor_id],
                    time_range=(timestamps[i], timestamps[i]),
                    parameters={'z_score': z_score, 'value': values[i], 'mean': mean_val, 'std': std_val},
                    impact_score=min(1.0, abs(z_score) / 4.0),
                    recommendations=[
                        "Investigate sensor calibration",
                        "Check for external events",
                        "Verify data collection integrity"
                    ],
                    metadata={'detection_method': 'z_score', 'threshold': self.thresholds['anomaly_z_score']}
                )
                patterns.append(pattern)

        return patterns

    async def _detect_trends(self, sensor_id: str, values: List[float], timestamps: List[datetime]) -> List[DetectedPattern]:
        """Detect trending patterns"""
        patterns = []

        if len(values) < self.thresholds['trend_min_points']:
            return patterns

        # Simple linear trend detection
        n = len(values)
        x = list(range(n))

        # Calculate correlation coefficient
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)

        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))

        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((values[i] - mean_y) ** 2 for i in range(n))

        if sum_sq_x == 0 or sum_sq_y == 0:
            return patterns

        correlation = numerator / math.sqrt(sum_sq_x * sum_sq_y)

        if abs(correlation) > self.thresholds['correlation_threshold']:
            trend_direction = "increasing" if correlation > 0 else "decreasing"

            pattern = DetectedPattern(
                pattern_id=f"trend_{sensor_id}_{len(self.detection_history)}",
                pattern_type=PatternType.TREND,
                confidence=abs(correlation),
                description=f"{trend_direction.capitalize()} trend detected (correlation: {correlation:.3f})",
                affected_sensors=[sensor_id],
                time_range=(timestamps[0], timestamps[-1]),
                parameters={'correlation': correlation, 'direction': trend_direction, 'slope': correlation},
                impact_score=abs(correlation),
                recommendations=[
                    f"Monitor {trend_direction} trend continuation",
                    "Analyze underlying causes",
                    "Consider intervention strategies"
                ],
                metadata={'detection_method': 'correlation', 'data_points': n}
            )
            patterns.append(pattern)

        return patterns

    async def _detect_cycles(self, sensor_id: str, values: List[float], timestamps: List[datetime]) -> List[DetectedPattern]:
        """Detect cyclical patterns (daily, weekly)"""
        patterns = []

        if len(values) < 24:  # Need at least a day's worth of data
            return patterns

        # Simple periodicity detection using autocorrelation
        patterns.extend(await self._detect_daily_cycle(sensor_id, values, timestamps))

        return patterns

    async def _detect_daily_cycle(self, sensor_id: str, values: List[float], timestamps: List[datetime]) -> List[DetectedPattern]:
        """Detect daily cyclical patterns"""
        patterns = []

        # Group by hour of day
        hourly_values = {}
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(values[i])

        # Check if we have values for multiple hours
        if len(hourly_values) < 6:
            return patterns

        # Calculate average for each hour
        hourly_averages = {}
        for hour, hour_values in hourly_values.items():
            hourly_averages[hour] = statistics.mean(hour_values)

        # Check for cyclical pattern (peak and valley)
        sorted_hours = sorted(hourly_averages.keys())
        hour_values = [hourly_averages[h] for h in sorted_hours]

        # Calculate variance to see if there's enough variation
        if len(hour_values) > 1:
            variance = statistics.variance(hour_values)
            mean_val = statistics.mean(hour_values)

            if variance > 0 and (variance / (mean_val ** 2)) > 0.1:  # Coefficient of variation > 0.1
                pattern = DetectedPattern(
                    pattern_id=f"daily_cycle_{sensor_id}_{len(self.detection_history)}",
                    pattern_type=PatternType.DAILY_CYCLE,
                    confidence=min(0.9, variance / (mean_val ** 2)),
                    description=f"Daily cyclical pattern detected with {len(hourly_averages)} hour samples",
                    affected_sensors=[sensor_id],
                    time_range=(timestamps[0], timestamps[-1]),
                    parameters={
                        'hourly_averages': hourly_averages,
                        'variance': variance,
                        'coefficient_of_variation': variance / (mean_val ** 2)
                    },
                    impact_score=min(1.0, variance / (mean_val ** 2)),
                    recommendations=[
                        "Optimize resource allocation based on daily patterns",
                        "Implement predictive scaling",
                        "Consider time-based interventions"
                    ],
                    metadata={'detection_method': 'hourly_variance', 'hours_analyzed': len(hourly_averages)}
                )
                patterns.append(pattern)

        return patterns

    async def analyze_correlations(self, sensor_data: Dict[str, List[Dict[str, Any]]]) -> List[DetectedPattern]:
        """Analyze correlations between different sensors"""
        patterns = []
        sensor_ids = list(sensor_data.keys())

        # Compare each pair of sensors
        for i in range(len(sensor_ids)):
            for j in range(i + 1, len(sensor_ids)):
                sensor1, sensor2 = sensor_ids[i], sensor_ids[j]

                values1 = [point.get('value', 0) for point in sensor_data[sensor1]]
                values2 = [point.get('value', 0) for point in sensor_data[sensor2]]

                # Ensure same length
                min_len = min(len(values1), len(values2))
                if min_len < 5:
                    continue

                values1 = values1[:min_len]
                values2 = values2[:min_len]

                # Calculate correlation
                correlation = self._calculate_correlation(values1, values2)

                if abs(correlation) > self.thresholds['correlation_threshold']:
                    relationship = "positive" if correlation > 0 else "negative"

                    pattern = DetectedPattern(
                        pattern_id=f"correlation_{sensor1}_{sensor2}",
                        pattern_type=PatternType.CORRELATION,
                        confidence=abs(correlation),
                        description=f"Strong {relationship} correlation between {sensor1} and {sensor2} (r={correlation:.3f})",
                        affected_sensors=[sensor1, sensor2],
                        time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                        parameters={'correlation': correlation, 'relationship': relationship},
                        impact_score=abs(correlation),
                        recommendations=[
                            "Investigate causal relationship",
                            "Consider joint optimization strategies",
                            "Monitor correlation stability"
                        ],
                        metadata={'detection_method': 'pearson_correlation', 'data_points': min_len}
                    )
                    patterns.append(pattern)

        return patterns

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        if sum_sq_x == 0 or sum_sq_y == 0:
            return 0.0

        return numerator / math.sqrt(sum_sq_x * sum_sq_y)

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all detected patterns"""
        pattern_types = {}
        total_patterns = len(self.detection_history)

        for pattern in self.detection_history:
            pattern_type = pattern.pattern_type.value
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1

        return {
            'total_patterns': total_patterns,
            'pattern_types': pattern_types,
            'latest_patterns': [
                {
                    'type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'description': p.description,
                    'sensors': p.affected_sensors
                }
                for p in self.detection_history[-10:]  # Last 10 patterns
            ]
        }