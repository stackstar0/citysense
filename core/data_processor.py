"""
Data Processor - Advanced urban data processing and storage
Handles real-time data streams, historical analysis, and data intelligence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import sqlite3
import statistics
import random
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Individual data point in the city intelligence system"""
    timestamp: datetime
    sensor_id: str
    sensor_type: str
    location: Dict[str, float]
    measurements: Dict[str, Any]
    quality_score: float  # 0-1, data quality assessment

@dataclass
class Pattern:
    """Detected pattern in city behavior"""
    pattern_id: str
    pattern_type: str  # "seasonal", "daily", "event-driven", "anomaly"
    description: str
    confidence: float
    affected_sensors: List[str]
    time_range: Tuple[datetime, datetime]
    parameters: Dict[str, Any]

class DataProcessor:
    """
    Advanced data processing engine for urban intelligence
    Handles real-time streams, storage, analysis, and pattern detection
    """

    def __init__(self, db_path: str = "data/citysense.db"):
        self.db_path = db_path
        self.status = "initializing"

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Data processing queues
        self._processing_queue = asyncio.Queue(maxsize=10000)
        self._batch_size = 100
        self._processing_interval = 10  # seconds

        # Cache for recent data
        self._recent_data_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_duration = timedelta(hours=1)

        self.status = "ready"

    def _init_database(self):
        """Initialize SQLite database for city data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Sensor readings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    sensor_id TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    measurements TEXT NOT NULL,  -- JSON blob
                    quality_score REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # City vital signs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vital_signs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    air_quality_index REAL,
                    energy_efficiency REAL,
                    traffic_flow_rate REAL,
                    economic_activity REAL,
                    ecological_health REAL,
                    social_wellbeing REAL,
                    resilience_score REAL,
                    stress_indicators TEXT,  -- JSON blob
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    affected_sensors TEXT,  -- JSON array
                    start_time DATETIME,
                    end_time DATETIME,
                    parameters TEXT,  -- JSON blob
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_id TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    predicted_impact TEXT,  -- JSON blob
                    recommended_actions TEXT,  -- JSON array
                    affected_areas TEXT,  -- JSON array
                    confidence_score REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id ON sensor_readings(sensor_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vital_signs_timestamp ON vital_signs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_timestamp ON insights(timestamp)")

            conn.commit()
            conn.close()

            logger.info("✅ Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def start(self):
        """Start data processing services"""
        logger.info("🚀 Starting data processing engine")
        self.status = "active"

        # Start background processing tasks
        asyncio.create_task(self._process_data_queue())
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._update_cache())

        logger.info("✅ Data processing engine active")

    async def ingest_sensor_data(self, sensor_id: str, sensor_type: str,
                                location: Dict[str, float], measurements: Dict[str, Any]):
        """Ingest new sensor data for processing"""
        try:
            # Calculate data quality score
            quality_score = self._assess_data_quality(measurements)

            data_point = DataPoint(
                timestamp=datetime.now(),
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                location=location,
                measurements=measurements,
                quality_score=quality_score
            )

            # Add to processing queue
            await self._processing_queue.put(data_point)

            # Update cache
            if sensor_id not in self._recent_data_cache:
                self._recent_data_cache[sensor_id] = []

            self._recent_data_cache[sensor_id].append({
                "timestamp": data_point.timestamp,
                "measurements": measurements,
                "quality_score": quality_score
            })

            # Keep cache size manageable
            if len(self._recent_data_cache[sensor_id]) > 1000:
                self._recent_data_cache[sensor_id] = self._recent_data_cache[sensor_id][-500:]

        except Exception as e:
            logger.error(f"Error ingesting sensor data: {e}")

    async def _process_data_queue(self):
        """Process queued sensor data in batches"""
        while True:
            try:
                batch = []

                # Collect batch of data points
                try:
                    # Wait for first item
                    data_point = await asyncio.wait_for(
                        self._processing_queue.get(), timeout=self._processing_interval
                    )
                    batch.append(data_point)

                    # Collect additional items up to batch size
                    while len(batch) < self._batch_size:
                        try:
                            data_point = self._processing_queue.get_nowait()
                            batch.append(data_point)
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    # No data received, continue loop
                    continue

                if batch:
                    await self._store_batch(batch)
                    logger.debug(f"Processed batch of {len(batch)} data points")

            except Exception as e:
                logger.error(f"Error in data processing queue: {e}")
                await asyncio.sleep(5)

    async def _store_batch(self, batch: List[DataPoint]):
        """Store batch of data points to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for data_point in batch:
                cursor.execute("""
                    INSERT INTO sensor_readings
                    (timestamp, sensor_id, sensor_type, latitude, longitude, elevation,
                     measurements, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_point.timestamp,
                    data_point.sensor_id,
                    data_point.sensor_type,
                    data_point.location.get("lat"),
                    data_point.location.get("lon"),
                    data_point.location.get("elevation"),
                    json.dumps(data_point.measurements),
                    data_point.quality_score
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing batch to database: {e}")

    def _assess_data_quality(self, measurements: Dict[str, Any]) -> float:
        """Assess quality of sensor measurements"""
        quality_factors = []

        # Check for completeness
        completeness = len([v for v in measurements.values() if v is not None]) / len(measurements)
        quality_factors.append(completeness)

        # Check for reasonable value ranges (simplified)
        range_checks = 0
        total_checks = 0

        for key, value in measurements.items():
            if isinstance(value, (int, float)):
                total_checks += 1

                # Basic range checks
                if key == "temperature_c" and -40 <= value <= 60:
                    range_checks += 1
                elif key == "humidity_percent" and 0 <= value <= 100:
                    range_checks += 1
                elif key == "pm25" and 0 <= value <= 500:
                    range_checks += 1
                elif key == "pm10" and 0 <= value <= 600:
                    range_checks += 1
                elif key.endswith("_kmh") and 0 <= value <= 200:
                    range_checks += 1
                elif 0 <= value <= 10000:  # Generic reasonable range
                    range_checks += 1

        if total_checks > 0:
            range_quality = range_checks / total_checks
            quality_factors.append(range_quality)

        # Check for timestamp freshness
        timestamp_quality = 1.0  # Assume fresh data for now
        quality_factors.append(timestamp_quality)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

    async def store_vital_signs(self, vital_signs):
        """Store city vital signs to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO vital_signs
                (timestamp, air_quality_index, energy_efficiency, traffic_flow_rate,
                 economic_activity, ecological_health, social_wellbeing, resilience_score,
                 stress_indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vital_signs.timestamp,
                vital_signs.air_quality_index,
                vital_signs.energy_efficiency,
                vital_signs.traffic_flow_rate,
                vital_signs.economic_activity,
                vital_signs.ecological_health,
                vital_signs.social_wellbeing,
                vital_signs.resilience_score,
                json.dumps(vital_signs.stress_indicators)
            ))

            conn.commit()
            conn.close()

            logger.debug("Stored city vital signs to database")

        except Exception as e:
            logger.error(f"Error storing vital signs: {e}")

    async def store_patterns(self, patterns: List[Pattern]):
        """Store detected patterns to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute("""
                    INSERT OR REPLACE INTO patterns
                    (pattern_id, pattern_type, description, confidence, affected_sensors,
                     start_time, end_time, parameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.description,
                    pattern.confidence,
                    json.dumps(pattern.affected_sensors),
                    pattern.time_range[0],
                    pattern.time_range[1],
                    json.dumps(pattern.parameters)
                ))

            conn.commit()
            conn.close()

            logger.debug(f"Stored {len(patterns)} patterns to database")

        except Exception as e:
            logger.error(f"Error storing patterns: {e}")

    async def get_historical_data(self, hours: int = 24,
                                 sensor_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get historical sensor data as pandas DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Build query
            query = """
                SELECT timestamp, sensor_id, sensor_type, latitude, longitude,
                       measurements, quality_score
                FROM sensor_readings
                WHERE timestamp >= datetime('now', '-{} hours')
            """.format(hours)

            if sensor_types:
                placeholders = ','.join(['?' for _ in sensor_types])
                query += f" AND sensor_type IN ({placeholders})"
                params = sensor_types
            else:
                params = []

            query += " ORDER BY timestamp DESC"

            cursor = conn.cursor()
            cursor.execute(query, params or [])
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Parse JSON measurements
                if 'measurements' in row_dict and isinstance(row_dict['measurements'], str):
                    try:
                        row_dict['measurements'] = json.loads(row_dict['measurements'])
                    except:
                        row_dict['measurements'] = {}
                # Convert timestamp
                if 'timestamp' in row_dict and isinstance(row_dict['timestamp'], str):
                    try:
                        row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                    except:
                        row_dict['timestamp'] = datetime.now()
                data.append(row_dict)

            return data

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []

    async def get_vital_signs_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical vital signs data"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT * FROM vital_signs
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)

            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert timestamp
                if 'timestamp' in row_dict and isinstance(row_dict['timestamp'], str):
                    try:
                        row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                    except:
                        row_dict['timestamp'] = datetime.now()
                # Parse JSON stress indicators
                if 'stress_indicators' in row_dict and isinstance(row_dict['stress_indicators'], str):
                    try:
                        row_dict['stress_indicators'] = json.loads(row_dict['stress_indicators'])
                    except:
                        row_dict['stress_indicators'] = {}
                data.append(row_dict)

            return data

        except Exception as e:
            logger.error(f"Error getting vital signs history: {e}")
            return []

    async def get_sensor_statistics(self, sensor_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical analysis for a specific sensor"""
        try:
            df = await self.get_historical_data(hours=hours)

            if df.empty:
                return {}

            sensor_data = df[df['sensor_id'] == sensor_id]

            if sensor_data.empty:
                return {}

            stats = {
                "sensor_id": sensor_id,
                "data_points": len(sensor_data),
                "time_range": {
                    "start": sensor_data['timestamp'].min().isoformat(),
                    "end": sensor_data['timestamp'].max().isoformat()
                },
                "average_quality_score": sensor_data['quality_score'].mean(),
                "measurements_stats": {}
            }

            # Calculate statistics for each measurement type
            all_measurements = {}
            for _, row in sensor_data.iterrows():
                for key, value in row['measurements'].items():
                    if isinstance(value, (int, float)):
                        if key not in all_measurements:
                            all_measurements[key] = []
                        all_measurements[key].append(value)

            for measurement, values in all_measurements.items():
                if values:
                    # Use basic statistics instead of pandas
                    sorted_values = sorted(values)
                    n = len(values)
                    median_val = sorted_values[n//2] if n % 2 == 1 else (sorted_values[n//2-1] + sorted_values[n//2]) / 2

                    stats["measurements_stats"][measurement] = {
                        "mean": statistics.mean(values),
                        "median": median_val,
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }

            return stats

        except Exception as e:
            logger.error(f"Error getting sensor statistics: {e}")
            return {}

    async def detect_anomalies(self, hours: int = 24, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in sensor data using statistical methods"""
        try:
            df = await self.get_historical_data(hours=hours)
            anomalies = []

            if df.empty:
                return anomalies

            # Group by sensor for anomaly detection
            for sensor_id in df['sensor_id'].unique():
                sensor_data = df[df['sensor_id'] == sensor_id]

                # Extract numeric measurements
                measurements_data = {}
                for _, row in sensor_data.iterrows():
                    for key, value in row['measurements'].items():
                        if isinstance(value, (int, float)):
                            if key not in measurements_data:
                                measurements_data[key] = []
                            measurements_data[key].append({
                                'timestamp': row['timestamp'],
                                'value': value
                            })

                # Detect anomalies using z-score
                for measurement, data_points in measurements_data.items():
                    if len(data_points) < 10:  # Need minimum data points
                        continue

                    values = [dp['value'] for dp in data_points]
                    mean_val = statistics.mean(values) if values else 0
                    std_val = statistics.stdev(values) if len(values) > 1 else 0

                    if std_val == 0:  # Avoid division by zero
                        continue

                    for dp in data_points:
                        z_score = abs((dp['value'] - mean_val) / std_val)

                        if z_score > threshold:
                            anomalies.append({
                                'sensor_id': sensor_id,
                                'measurement': measurement,
                                'timestamp': dp['timestamp'].isoformat(),
                                'value': dp['value'],
                                'expected_range': {
                                    'mean': mean_val,
                                    'std': std_val
                                },
                                'z_score': z_score,
                                'anomaly_score': min(1.0, z_score / 5.0)  # Normalize to 0-1
                            })

            # Sort by anomaly score (highest first)
            anomalies.sort(key=lambda x: x['z_score'], reverse=True)

            return anomalies[:50]  # Return top 50 anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []

    async def _cleanup_old_data(self):
        """Clean up old data to manage database size"""
        while True:
            try:
                # Clean up data older than 30 days
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Delete old sensor readings
                cursor.execute("""
                    DELETE FROM sensor_readings
                    WHERE timestamp < datetime('now', '-30 days')
                """)

                # Delete old vital signs
                cursor.execute("""
                    DELETE FROM vital_signs
                    WHERE timestamp < datetime('now', '-30 days')
                """)

                # Delete old patterns
                cursor.execute("""
                    DELETE FROM patterns
                    WHERE end_time < datetime('now', '-7 days')
                """)

                # Delete old insights
                cursor.execute("""
                    DELETE FROM insights
                    WHERE timestamp < datetime('now', '-7 days')
                """)

                conn.commit()
                conn.close()

                logger.debug("Cleaned up old data from database")

            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")

            # Run cleanup daily
            await asyncio.sleep(24 * 3600)

    async def _update_cache(self):
        """Update recent data cache"""
        while True:
            try:
                # Clean old entries from cache
                cutoff_time = datetime.now() - self._cache_duration

                for sensor_id, data_list in self._recent_data_cache.items():
                    self._recent_data_cache[sensor_id] = [
                        dp for dp in data_list
                        if dp['timestamp'] > cutoff_time
                    ]

            except Exception as e:
                logger.error(f"Error updating cache: {e}")

            await asyncio.sleep(300)  # Update every 5 minutes

    async def get_system_status(self) -> Dict[str, Any]:
        """Get data processing system status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get record counts
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            sensor_readings_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM vital_signs")
            vital_signs_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns")
            patterns_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM insights")
            insights_count = cursor.fetchone()[0]

            # Get latest data timestamp
            cursor.execute("SELECT MAX(timestamp) FROM sensor_readings")
            latest_reading = cursor.fetchone()[0]

            conn.close()

            return {
                "status": self.status,
                "database": {
                    "sensor_readings": sensor_readings_count,
                    "vital_signs": vital_signs_count,
                    "patterns": patterns_count,
                    "insights": insights_count
                },
                "processing": {
                    "queue_size": self._processing_queue.qsize(),
                    "cache_sensors": len(self._recent_data_cache),
                    "latest_reading": latest_reading
                },
                "performance": {
                    "batch_size": self._batch_size,
                    "processing_interval": self._processing_interval
                }
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "error": str(e)}