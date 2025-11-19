"""
Sensor Manager - Coordinates urban sensing network
Manages IoT sensors, data collection, and real-time monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json
import random
from enum import Enum

logger = logging.getLogger(__name__)

class SensorType(Enum):
    AIR_QUALITY = "air_quality"
    TRAFFIC = "traffic"
    ENERGY = "energy"
    NOISE = "noise"
    WEATHER = "weather"
    WATER_QUALITY = "water_quality"
    WASTE = "waste"
    PEDESTRIAN = "pedestrian"
    ECONOMIC = "economic"
    SOCIAL = "social"

class SensorStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class Sensor:
    """Individual sensor configuration and state"""
    sensor_id: str
    sensor_type: SensorType
    location: Dict[str, float]  # {"lat": float, "lon": float, "elevation": float}
    status: SensorStatus
    last_reading: Optional[Dict[str, Any]]
    last_update: Optional[datetime]
    metadata: Dict[str, Any]

class SensorManager:
    """
    Manages the entire urban sensor network
    Coordinates data collection, sensor health, and real-time monitoring
    """

    def __init__(self):
        self.sensors: Dict[str, Sensor] = {}
        self.data_callbacks: List[Callable] = []
        self.status = "initializing"
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Initialize sensor network
        self._initialize_sensor_network()

    def _initialize_sensor_network(self):
        """Initialize the urban sensor network"""
        logger.info("🔧 Initializing urban sensor network")

        # Air Quality Sensors (distributed across city)
        air_locations = [
            {"lat": 40.7128, "lon": -74.0060, "elevation": 10},  # Downtown
            {"lat": 40.7589, "lon": -73.9851, "elevation": 15},  # Midtown
            {"lat": 40.6782, "lon": -73.9442, "elevation": 8},   # Brooklyn
            {"lat": 40.7505, "lon": -73.9934, "elevation": 20},  # Industrial
        ]

        for i, location in enumerate(air_locations):
            sensor_id = f"air_quality_{i+1}"
            self.sensors[sensor_id] = Sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.AIR_QUALITY,
                location=location,
                status=SensorStatus.ACTIVE,
                last_reading=None,
                last_update=None,
                metadata={"model": "AQM-3000", "calibration_date": "2024-01-15"}
            )

        # Traffic Sensors (major intersections and highways)
        traffic_locations = [
            {"lat": 40.7500, "lon": -73.9850, "elevation": 5},   # Times Square
            {"lat": 40.7282, "lon": -74.0776, "elevation": 3},   # Holland Tunnel
            {"lat": 40.7831, "lon": -73.9712, "elevation": 8},   # Central Park South
            {"lat": 40.6892, "lon": -74.0445, "elevation": 2},   # Brooklyn Bridge
        ]

        for i, location in enumerate(traffic_locations):
            sensor_id = f"traffic_{i+1}"
            self.sensors[sensor_id] = Sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.TRAFFIC,
                location=location,
                status=SensorStatus.ACTIVE,
                last_reading=None,
                last_update=None,
                metadata={"type": "traffic_camera_ai", "lanes_monitored": 4}
            )

        # Energy Sensors (power grid monitoring)
        energy_locations = [
            {"lat": 40.7580, "lon": -73.9855, "elevation": 0},   # Main Grid Station
            {"lat": 40.7128, "lon": -74.0060, "elevation": 0},   # Downtown Substation
            {"lat": 40.6782, "lon": -73.9442, "elevation": 0},   # Brooklyn Grid
        ]

        for i, location in enumerate(energy_locations):
            sensor_id = f"energy_{i+1}"
            self.sensors[sensor_id] = Sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.ENERGY,
                location=location,
                status=SensorStatus.ACTIVE,
                last_reading=None,
                last_update=None,
                metadata={"capacity_mw": 500, "renewable_connected": True}
            )

        # Weather Stations
        weather_locations = [
            {"lat": 40.7589, "lon": -73.9851, "elevation": 25},  # Central Station
            {"lat": 40.6892, "lon": -74.0445, "elevation": 5},   # Waterfront
        ]

        for i, location in enumerate(weather_locations):
            sensor_id = f"weather_{i+1}"
            self.sensors[sensor_id] = Sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.WEATHER,
                location=location,
                status=SensorStatus.ACTIVE,
                last_reading=None,
                last_update=None,
                metadata={"model": "WS-5000", "wind_measurement": True}
            )

        # Water Quality Sensors
        water_locations = [
            {"lat": 40.7282, "lon": -74.0776, "elevation": -2},  # Hudson River
            {"lat": 40.6892, "lon": -74.0445, "elevation": -1},  # East River
        ]

        for i, location in enumerate(water_locations):
            sensor_id = f"water_{i+1}"
            self.sensors[sensor_id] = Sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.WATER_QUALITY,
                location=location,
                status=SensorStatus.ACTIVE,
                last_reading=None,
                last_update=None,
                metadata={"depth_m": 2, "parameters": ["ph", "dissolved_oxygen", "turbidity"]}
            )

        logger.info(f"✅ Initialized {len(self.sensors)} sensors across the urban network")

    async def start(self):
        """Start sensor data collection"""
        logger.info("🚀 Starting urban sensor network")
        self._running = True
        self.status = "active"

        # Start sensor reading tasks
        self._tasks = [
            asyncio.create_task(self._collect_sensor_data()),
            asyncio.create_task(self._monitor_sensor_health()),
            asyncio.create_task(self._simulate_real_time_data())  # For demo purposes
        ]

        logger.info("✅ Urban sensor network is now active")

    async def stop(self):
        """Stop sensor data collection"""
        logger.info("⏹ Stopping urban sensor network")
        self._running = False
        self.status = "offline"

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("✅ Urban sensor network stopped")

    async def _collect_sensor_data(self):
        """Main sensor data collection loop"""
        while self._running:
            try:
                for sensor_id, sensor in self.sensors.items():
                    if sensor.status == SensorStatus.ACTIVE:
                        # Collect data from sensor
                        reading = await self._read_sensor(sensor)

                        if reading:
                            sensor.last_reading = reading
                            sensor.last_update = datetime.now()

                            # Notify callbacks
                            for callback in self.data_callbacks:
                                try:
                                    await callback(sensor_id, reading)
                                except Exception as e:
                                    logger.error(f"Error in data callback: {e}")

            except Exception as e:
                logger.error(f"Error in sensor data collection: {e}")

            await asyncio.sleep(10)  # Collect data every 10 seconds

    async def _monitor_sensor_health(self):
        """Monitor sensor health and connectivity"""
        while self._running:
            try:
                current_time = datetime.now()

                for sensor_id, sensor in self.sensors.items():
                    # Check if sensor has been silent too long
                    if sensor.last_update:
                        time_since_update = current_time - sensor.last_update

                        if time_since_update > timedelta(minutes=5):
                            if sensor.status == SensorStatus.ACTIVE:
                                logger.warning(f"Sensor {sensor_id} may be unresponsive")
                                sensor.status = SensorStatus.ERROR

                    # Simulate occasional maintenance
                    if random.random() < 0.001:  # 0.1% chance per check
                        if sensor.status == SensorStatus.ACTIVE:
                            sensor.status = SensorStatus.MAINTENANCE
                            logger.info(f"Sensor {sensor_id} entering maintenance mode")

                    # Recover from maintenance
                    elif sensor.status == SensorStatus.MAINTENANCE and random.random() < 0.1:
                        sensor.status = SensorStatus.ACTIVE
                        logger.info(f"Sensor {sensor_id} back online from maintenance")

            except Exception as e:
                logger.error(f"Error in sensor health monitoring: {e}")

            await asyncio.sleep(60)  # Check health every minute

    async def _simulate_real_time_data(self):
        """Simulate realistic sensor data for demonstration"""
        while self._running:
            try:
                current_time = datetime.now()
                hour = current_time.hour

                # Simulate daily patterns
                for sensor in self.sensors.values():
                    if sensor.status != SensorStatus.ACTIVE:
                        continue

                    base_reading = await self._generate_realistic_reading(sensor, hour)

                    # Add some realistic noise and trends
                    if sensor.sensor_type == SensorType.AIR_QUALITY:
                        # Worse air quality during rush hours
                        rush_hour_factor = 1.3 if hour in [8, 9, 17, 18, 19] else 1.0
                        base_reading = self._adjust_reading(base_reading, rush_hour_factor)

                    elif sensor.sensor_type == SensorType.TRAFFIC:
                        # Higher traffic during business hours
                        business_factor = 1.5 if 7 <= hour <= 19 else 0.6
                        base_reading = self._adjust_reading(base_reading, business_factor)

                    elif sensor.sensor_type == SensorType.ENERGY:
                        # Higher energy consumption during day and evening
                        energy_factor = 1.4 if 8 <= hour <= 22 else 0.8
                        base_reading = self._adjust_reading(base_reading, energy_factor)

                    sensor.last_reading = base_reading
                    sensor.last_update = current_time

            except Exception as e:
                logger.error(f"Error in data simulation: {e}")

            await asyncio.sleep(30)  # Update every 30 seconds

    async def _read_sensor(self, sensor: Sensor) -> Optional[Dict[str, Any]]:
        """Read data from a specific sensor"""
        try:
            # In a real implementation, this would interface with actual sensors
            # For now, we'll generate realistic simulated data
            return await self._generate_realistic_reading(sensor, datetime.now().hour)

        except Exception as e:
            logger.error(f"Error reading sensor {sensor.sensor_id}: {e}")
            return None

    async def _generate_realistic_reading(self, sensor: Sensor, hour: int) -> Dict[str, Any]:
        """Generate realistic sensor readings based on sensor type and time"""
        base_time = datetime.now()

        if sensor.sensor_type == SensorType.AIR_QUALITY:
            return {
                "timestamp": base_time.isoformat(),
                "pm25": random.uniform(8, 35) + (5 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0),
                "pm10": random.uniform(15, 50) + (8 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0),
                "no2": random.uniform(20, 60) + (10 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0),
                "co": random.uniform(1, 8),
                "o3": random.uniform(30, 120),
                "temperature": random.uniform(15, 25),
                "humidity": random.uniform(40, 80)
            }

        elif sensor.sensor_type == SensorType.TRAFFIC:
            base_speed = 45 if hour < 7 or hour > 20 else 25
            return {
                "timestamp": base_time.isoformat(),
                "vehicle_count": random.randint(20, 200) if 7 <= hour <= 19 else random.randint(5, 50),
                "average_speed_kmh": base_speed + random.uniform(-10, 10),
                "congestion_level": random.uniform(0.1, 0.8) if 7 <= hour <= 19 else random.uniform(0.0, 0.3),
                "pedestrian_count": random.randint(10, 100),
                "bicycle_count": random.randint(2, 30)
            }

        elif sensor.sensor_type == SensorType.ENERGY:
            base_consumption = 400 if 8 <= hour <= 22 else 250
            return {
                "timestamp": base_time.isoformat(),
                "power_consumption_mw": base_consumption + random.uniform(-50, 100),
                "renewable_generation_mw": random.uniform(50, 200),
                "grid_frequency_hz": 60.0 + random.uniform(-0.1, 0.1),
                "voltage_kv": 13.8 + random.uniform(-0.5, 0.5),
                "power_factor": random.uniform(0.85, 0.95)
            }

        elif sensor.sensor_type == SensorType.WEATHER:
            return {
                "timestamp": base_time.isoformat(),
                "temperature_c": random.uniform(10, 30),
                "humidity_percent": random.uniform(30, 90),
                "wind_speed_kmh": random.uniform(0, 25),
                "wind_direction_deg": random.uniform(0, 360),
                "pressure_hpa": random.uniform(1000, 1030),
                "precipitation_mm": random.uniform(0, 2) if random.random() < 0.1 else 0,
                "uv_index": max(0, random.uniform(0, 10) if 6 <= hour <= 18 else 0)
            }

        elif sensor.sensor_type == SensorType.WATER_QUALITY:
            return {
                "timestamp": base_time.isoformat(),
                "ph": random.uniform(6.5, 8.5),
                "dissolved_oxygen_mg_l": random.uniform(5, 12),
                "turbidity_ntu": random.uniform(1, 10),
                "temperature_c": random.uniform(8, 25),
                "conductivity_us_cm": random.uniform(200, 800),
                "nitrates_mg_l": random.uniform(0.1, 5.0),
                "phosphates_mg_l": random.uniform(0.01, 0.5)
            }

        else:
            return {
                "timestamp": base_time.isoformat(),
                "value": random.uniform(0, 100),
                "status": "normal"
            }

    def _adjust_reading(self, reading: Dict[str, Any], factor: float) -> Dict[str, Any]:
        """Adjust sensor reading values by a factor"""
        adjusted = reading.copy()

        for key, value in reading.items():
            if key != "timestamp" and isinstance(value, (int, float)):
                # Apply factor with some randomness
                adjusted[key] = value * factor * random.uniform(0.9, 1.1)

        return adjusted

    async def get_latest_readings(self) -> Dict[str, Any]:
        """Get latest readings from all active sensors"""
        readings = {
            "timestamp": datetime.now().isoformat(),
            "sensors_active": len([s for s in self.sensors.values() if s.status == SensorStatus.ACTIVE]),
            "sensors_total": len(self.sensors)
        }

        # Group readings by sensor type
        for sensor_type in SensorType:
            type_sensors = [s for s in self.sensors.values()
                          if s.sensor_type == sensor_type and s.status == SensorStatus.ACTIVE]

            if type_sensors:
                type_readings = []
                for sensor in type_sensors:
                    if sensor.last_reading:
                        reading = sensor.last_reading.copy()
                        reading["sensor_id"] = sensor.sensor_id
                        reading["location"] = sensor.location
                        type_readings.append(reading)

                if type_readings:
                    readings[f"{sensor_type.value}_sensors"] = type_readings

        # Calculate aggregated metrics
        readings.update(await self._calculate_aggregated_metrics(readings))

        return readings

    async def _calculate_aggregated_metrics(self, readings: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate city-wide aggregated metrics"""
        aggregated = {}

        # Air quality aggregation
        if "air_quality_sensors" in readings:
            air_readings = readings["air_quality_sensors"]
            if air_readings:
                avg_pm25 = sum(r.get("pm25", 0) for r in air_readings) / len(air_readings)
                avg_pm10 = sum(r.get("pm10", 0) for r in air_readings) / len(air_readings)
                avg_no2 = sum(r.get("no2", 0) for r in air_readings) / len(air_readings)

                aggregated["air_quality"] = {
                    "average_pm25": avg_pm25,
                    "average_pm10": avg_pm10,
                    "average_no2": avg_no2,
                    "air_quality_index": max(0, 100 - (avg_pm25/35 + avg_pm10/50 + avg_no2/40) * 33.33)
                }

        # Traffic aggregation
        if "traffic_sensors" in readings:
            traffic_readings = readings["traffic_sensors"]
            if traffic_readings:
                total_vehicles = sum(r.get("vehicle_count", 0) for r in traffic_readings)
                avg_speed = sum(r.get("average_speed_kmh", 0) for r in traffic_readings) / len(traffic_readings)
                avg_congestion = sum(r.get("congestion_level", 0) for r in traffic_readings) / len(traffic_readings)

                aggregated["traffic_summary"] = {
                    "total_vehicles_monitored": total_vehicles,
                    "average_speed_kmh": avg_speed,
                    "congestion_ratio": avg_congestion,
                    "traffic_flow_efficiency": max(0, 100 - avg_congestion * 100)
                }

        # Energy aggregation
        if "energy_sensors" in readings:
            energy_readings = readings["energy_sensors"]
            if energy_readings:
                total_consumption = sum(r.get("power_consumption_mw", 0) for r in energy_readings)
                total_renewable = sum(r.get("renewable_generation_mw", 0) for r in energy_readings)

                aggregated["energy_summary"] = {
                    "total_consumption_mw": total_consumption,
                    "renewable_generation_mw": total_renewable,
                    "renewable_ratio": total_renewable / max(total_consumption, 1),
                    "grid_efficiency": random.uniform(0.85, 0.95)  # Simulated
                }

        return aggregated

    async def adjust_traffic_signals(self, parameters: Dict[str, Any]):
        """Adjust traffic signal timing for optimization"""
        logger.info(f"Adjusting traffic signals: {parameters}")
        # In real implementation, would interface with traffic control systems

    async def adjust_energy_distribution(self, parameters: Dict[str, Any]):
        """Adjust energy distribution for load balancing"""
        logger.info(f"Adjusting energy distribution: {parameters}")
        # In real implementation, would interface with smart grid systems

    async def activate_air_purification(self, parameters: Dict[str, Any]):
        """Activate air purification systems"""
        logger.info(f"Activating air purification: {parameters}")
        # In real implementation, would control air purification infrastructure

    def add_data_callback(self, callback: Callable):
        """Add callback for real-time sensor data"""
        self.data_callbacks.append(callback)

    def get_sensor_status(self) -> Dict[str, Any]:
        """Get current sensor network status"""
        status_counts = {}
        for status in SensorStatus:
            status_counts[status.value] = len([s for s in self.sensors.values() if s.status == status])

        return {
            "total_sensors": len(self.sensors),
            "status_breakdown": status_counts,
            "network_health": status_counts.get("active", 0) / len(self.sensors) * 100,
            "last_update": max([s.last_update for s in self.sensors.values() if s.last_update],
                              default=datetime.now()).isoformat()
        }