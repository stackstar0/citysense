"""
Real-time Environmental System
Integrates IoT sensors and weather APIs for live environmental monitoring
and climate-responsive design recommendations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import math
import statistics

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_id: str
    sensor_type: str
    location: Dict[str, float]  # lat, lon
    timestamp: datetime
    value: float
    unit: str
    quality_score: float
    metadata: Dict[str, Any]

@dataclass
class WeatherData:
    """Weather API data structure"""
    location: str
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    precipitation: float
    cloud_cover: float
    visibility: float
    uv_index: float
    air_quality_index: Optional[float]

@dataclass
class EnvironmentalMetrics:
    """Comprehensive environmental metrics"""
    timestamp: datetime
    location: Dict[str, float]
    air_quality: Dict[str, float]
    climate_comfort: Dict[str, float]
    energy_efficiency: Dict[str, float]
    water_metrics: Dict[str, float]
    biodiversity_indicators: Dict[str, float]
    urban_heat_island: float
    carbon_metrics: Dict[str, float]

@dataclass
class ClimateRecommendation:
    """Climate-responsive design recommendation"""
    recommendation_id: str
    category: str
    title: str
    description: str
    priority: str  # high, medium, low
    impact_score: float
    implementation_timeframe: str
    cost_estimate: str
    environmental_triggers: List[str]
    design_principles: List[str]
    technical_specifications: Dict[str, Any]

class RealTimeEnvironmentalSystem:
    """
    Comprehensive real-time environmental monitoring system
    Integrates IoT sensors, weather APIs, and AI for climate-responsive recommendations
    """

    def __init__(self):
        self.iot_sensors = {}
        self.weather_data = None
        self.environmental_metrics = None
        self.sensor_history = {}
        self.weather_history = []
        self.recommendations = []
        self.alert_thresholds = self._initialize_thresholds()
        self.update_interval = 30  # 30 seconds for demo
        self.running = False

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for environmental parameters"""
        return {
            'air_quality': {
                'pm25_critical': 75.0,
                'pm10_critical': 150.0,
                'no2_critical': 200.0,
                'o3_critical': 180.0,
                'co_critical': 30.0
            },
            'temperature': {
                'heat_warning': 35.0,
                'cold_warning': -10.0,
                'comfort_min': 18.0,
                'comfort_max': 26.0
            },
            'humidity': {
                'low_threshold': 30.0,
                'high_threshold': 70.0,
                'comfort_min': 40.0,
                'comfort_max': 60.0
            },
            'wind': {
                'strong_wind': 15.0,  # m/s
                'extreme_wind': 25.0
            },
            'precipitation': {
                'heavy_rain': 10.0,  # mm/h
                'extreme_rain': 25.0
            }
        }

    async def initialize(self):
        """Initialize the environmental monitoring system"""
        logger.info("🌍 Initializing Real-Time Environmental System...")

        try:
            # Initialize IoT sensor network
            await self._initialize_iot_sensors()

            # Initialize weather API connections
            await self._initialize_weather_apis()

            # Set running state
            self.running = True

            logger.info("✅ Environmental system initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize environmental system: {e}")
            # Don't raise - allow system to continue with basic functionality
            self.running = True

    async def _initialize_iot_sensors(self):
        """Initialize IoT sensor network"""
        # Simulated IoT sensors with realistic locations
        self.iot_sensors = {
            'air_quality_downtown': {
                'type': 'air_quality',
                'location': {'lat': 40.7589, 'lon': -73.9851},
                'sensors': ['pm25', 'pm10', 'no2', 'o3', 'co'],
                'status': 'active',
                'last_reading': None
            },
            'weather_station_01': {
                'type': 'weather',
                'location': {'lat': 40.7614, 'lon': -73.9776},
                'sensors': ['temperature', 'humidity', 'pressure', 'wind'],
                'status': 'active',
                'last_reading': None
            },
            'energy_meter_residential': {
                'type': 'energy',
                'location': {'lat': 40.7505, 'lon': -73.9934},
                'sensors': ['power_consumption', 'solar_generation'],
                'status': 'active',
                'last_reading': None
            },
            'water_quality_01': {
                'type': 'water',
                'location': {'lat': 40.7549, 'lon': -73.9840},
                'sensors': ['ph', 'turbidity', 'dissolved_oxygen', 'temperature'],
                'status': 'active',
                'last_reading': None
            },
            'noise_monitor_commercial': {
                'type': 'noise',
                'location': {'lat': 40.7580, 'lon': -73.9855},
                'sensors': ['decibel_level', 'frequency_analysis'],
                'status': 'active',
                'last_reading': None
            },
            'biodiversity_cam_park': {
                'type': 'biodiversity',
                'location': {'lat': 40.7829, 'lon': -73.9654},
                'sensors': ['species_count', 'vegetation_health'],
                'status': 'active',
                'last_reading': None
            }
        }

    async def _initialize_weather_apis(self):
        """Initialize weather API connections"""
        # In production, you would use real APIs like OpenWeatherMap
        # For now, we'll simulate realistic weather data
        logger.info("🌤️ Initializing weather API connections...")

    async def _data_collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                # Collect IoT sensor data
                await self._collect_iot_data()

                # Collect weather data
                await self._collect_weather_data()

                # Process and analyze data
                await self._process_environmental_data()

                # Generate recommendations
                await self._generate_climate_recommendations()

                # Check for alerts
                await self._check_environmental_alerts()

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _collect_iot_data(self):
        """Collect data from IoT sensors"""
        current_time = datetime.now()

        for sensor_id, sensor_config in self.iot_sensors.items():
            try:
                # Simulate realistic sensor readings
                readings = await self._simulate_sensor_readings(sensor_id, sensor_config)

                # Store readings with history
                if sensor_id not in self.sensor_history:
                    self.sensor_history[sensor_id] = []

                self.sensor_history[sensor_id].extend(readings)

                # Keep only last 24 hours of data
                cutoff_time = current_time - timedelta(hours=24)
                self.sensor_history[sensor_id] = [
                    r for r in self.sensor_history[sensor_id]
                    if r.timestamp > cutoff_time
                ]

                sensor_config['last_reading'] = current_time

            except Exception as e:
                logger.error(f"Error collecting data from sensor {sensor_id}: {e}")
                sensor_config['status'] = 'error'

    async def _simulate_sensor_readings(self, sensor_id: str, config: Dict) -> List[SensorReading]:
        """Simulate realistic sensor readings"""
        readings = []
        current_time = datetime.now()
        base_hour = current_time.hour

        # Simulate readings based on sensor type
        if config['type'] == 'air_quality':
            readings.extend([
                SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='pm25',
                    location=config['location'],
                    timestamp=current_time,
                    value=self._simulate_pm25(base_hour),
                    unit='μg/m³',
                    quality_score=0.85 + 0.1 * (hash(sensor_id) % 10) / 10,
                    metadata={'calibration_date': '2024-11-01'}
                ),
                SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='no2',
                    location=config['location'],
                    timestamp=current_time,
                    value=self._simulate_no2(base_hour),
                    unit='μg/m³',
                    quality_score=0.87 + 0.1 * (hash(sensor_id + 'no2') % 10) / 10,
                    metadata={'calibration_date': '2024-11-01'}
                )
            ])

        elif config['type'] == 'weather':
            readings.extend([
                SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='temperature',
                    location=config['location'],
                    timestamp=current_time,
                    value=self._simulate_temperature(base_hour),
                    unit='°C',
                    quality_score=0.95,
                    metadata={'sensor_model': 'DHT22'}
                ),
                SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='humidity',
                    location=config['location'],
                    timestamp=current_time,
                    value=self._simulate_humidity(base_hour),
                    unit='%',
                    quality_score=0.93,
                    metadata={'sensor_model': 'DHT22'}
                )
            ])

        elif config['type'] == 'energy':
            readings.append(
                SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='power_consumption',
                    location=config['location'],
                    timestamp=current_time,
                    value=self._simulate_energy_consumption(base_hour),
                    unit='kWh',
                    quality_score=0.98,
                    metadata={'meter_type': 'smart_meter'}
                )
            )

        return readings

    def _simulate_pm25(self, hour: int) -> float:
        """Simulate realistic PM2.5 readings with daily patterns"""
        # Higher in morning/evening traffic, lower midday
        base_value = 25.0
        traffic_factor = 1.5 if hour in [7, 8, 17, 18, 19] else 1.0
        random_variation = (hash(f"pm25_{hour}_{time.time()}") % 100) / 100 * 0.3
        return max(5.0, base_value * traffic_factor + random_variation * 10)

    def _simulate_no2(self, hour: int) -> float:
        """Simulate NO2 levels"""
        base_value = 40.0
        traffic_factor = 1.8 if hour in [7, 8, 17, 18, 19] else 1.0
        random_variation = (hash(f"no2_{hour}_{time.time()}") % 100) / 100 * 0.2
        return max(10.0, base_value * traffic_factor + random_variation * 15)

    def _simulate_temperature(self, hour: int) -> float:
        """Simulate temperature with daily cycle"""
        # Sinusoidal daily temperature pattern
        base_temp = 15.0  # November average
        daily_amplitude = 8.0
        temp = base_temp + daily_amplitude * math.sin((hour - 6) * math.pi / 12)
        random_variation = (hash(f"temp_{hour}_{time.time()}") % 100) / 100 * 2 - 1
        return temp + random_variation

    def _simulate_humidity(self, hour: int) -> float:
        """Simulate humidity levels"""
        # Inverse relationship with temperature generally
        base_humidity = 65.0
        daily_variation = -15.0 * math.sin((hour - 6) * math.pi / 12)
        random_variation = (hash(f"humidity_{hour}_{time.time()}") % 100) / 100 * 10 - 5
        return max(20.0, min(95.0, base_humidity + daily_variation + random_variation))

    def _simulate_energy_consumption(self, hour: int) -> float:
        """Simulate energy consumption patterns"""
        # Higher consumption in evening, lower at night
        base_consumption = 2.5
        time_factor = 1.0
        if 6 <= hour <= 9:  # Morning peak
            time_factor = 1.4
        elif 17 <= hour <= 22:  # Evening peak
            time_factor = 1.8
        elif 23 <= hour or hour <= 5:  # Night low
            time_factor = 0.6

        random_variation = (hash(f"energy_{hour}_{time.time()}") % 100) / 100 * 0.3
        return base_consumption * time_factor + random_variation

    async def _collect_weather_data(self):
        """Collect weather data from APIs"""
        # Simulate weather API data
        current_time = datetime.now()
        hour = current_time.hour

        self.weather_data = WeatherData(
            location="New York City",
            timestamp=current_time,
            temperature=self._simulate_temperature(hour),
            humidity=self._simulate_humidity(hour),
            pressure=1013.25 + (hash(f"pressure_{time.time()}") % 20) - 10,
            wind_speed=max(0, 5 + (hash(f"wind_{time.time()}") % 10) - 5),
            wind_direction=(hash(f"wind_dir_{time.time()}") % 360),
            precipitation=max(0, (hash(f"rain_{time.time()}") % 100) / 100 * 2 - 1.8),
            cloud_cover=(hash(f"clouds_{time.time()}") % 100),
            visibility=max(1, 10 + (hash(f"vis_{time.time()}") % 10) - 5),
            uv_index=max(0, min(11, 5 + (hash(f"uv_{time.time()}") % 6) - 3)),
            air_quality_index=self._calculate_aqi()
        )

        # Store in history
        self.weather_history.append(self.weather_data)

        # Keep only last 7 days
        cutoff_time = current_time - timedelta(days=7)
        self.weather_history = [
            w for w in self.weather_history
            if w.timestamp > cutoff_time
        ]

    def _calculate_aqi(self) -> float:
        """Calculate Air Quality Index from sensor data"""
        if not self.sensor_history:
            return 50.0  # Default moderate AQI

        # Get latest PM2.5 readings
        pm25_values = []
        for sensor_data in self.sensor_history.values():
            for reading in sensor_data[-10:]:  # Last 10 readings
                if reading.sensor_type == 'pm25':
                    pm25_values.append(reading.value)

        if not pm25_values:
            return 50.0

        avg_pm25 = statistics.mean(pm25_values)

        # Convert PM2.5 to AQI (simplified EPA formula)
        if avg_pm25 <= 12.0:
            aqi = avg_pm25 * 50 / 12.0
        elif avg_pm25 <= 35.4:
            aqi = 50 + (avg_pm25 - 12.0) * 50 / (35.4 - 12.0)
        elif avg_pm25 <= 55.4:
            aqi = 100 + (avg_pm25 - 35.4) * 50 / (55.4 - 35.4)
        else:
            aqi = min(300, 150 + (avg_pm25 - 55.4) * 150 / (150.4 - 55.4))

        return round(aqi, 1)

    async def _process_environmental_data(self):
        """Process collected data into comprehensive metrics"""
        if not self.weather_data:
            return

        current_time = datetime.now()

        # Calculate air quality metrics
        air_quality = self._calculate_air_quality_metrics()

        # Calculate climate comfort
        climate_comfort = self._calculate_climate_comfort()

        # Calculate energy efficiency
        energy_efficiency = self._calculate_energy_efficiency()

        # Calculate water metrics
        water_metrics = self._calculate_water_metrics()

        # Calculate biodiversity indicators
        biodiversity = self._calculate_biodiversity_indicators()

        # Calculate urban heat island effect
        urban_heat_island = self._calculate_urban_heat_island()

        # Calculate carbon metrics
        carbon_metrics = self._calculate_carbon_metrics()

        self.environmental_metrics = EnvironmentalMetrics(
            timestamp=current_time,
            location={'lat': 40.7589, 'lon': -73.9851},
            air_quality=air_quality,
            climate_comfort=climate_comfort,
            energy_efficiency=energy_efficiency,
            water_metrics=water_metrics,
            biodiversity_indicators=biodiversity,
            urban_heat_island=urban_heat_island,
            carbon_metrics=carbon_metrics
        )

    def _calculate_air_quality_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive air quality metrics"""
        metrics = {
            'overall_aqi': self.weather_data.air_quality_index or 50.0,
            'pm25_level': 0.0,
            'no2_level': 0.0,
            'health_risk': 'low'
        }

        # Get latest sensor readings
        for sensor_data in self.sensor_history.values():
            for reading in sensor_data[-5:]:  # Last 5 readings
                if reading.sensor_type == 'pm25':
                    metrics['pm25_level'] = reading.value
                elif reading.sensor_type == 'no2':
                    metrics['no2_level'] = reading.value

        # Determine health risk
        aqi = metrics['overall_aqi']
        if aqi <= 50:
            metrics['health_risk'] = 'low'
        elif aqi <= 100:
            metrics['health_risk'] = 'moderate'
        elif aqi <= 150:
            metrics['health_risk'] = 'unhealthy_sensitive'
        else:
            metrics['health_risk'] = 'unhealthy'

        return metrics

    def _calculate_climate_comfort(self) -> Dict[str, float]:
        """Calculate climate comfort indices"""
        temp = self.weather_data.temperature
        humidity = self.weather_data.humidity
        wind_speed = self.weather_data.wind_speed

        # Heat index calculation (simplified)
        heat_index = temp
        if temp >= 27 and humidity >= 40:
            heat_index = temp + 0.5 * (humidity - 40) * 0.1

        # Comfort index (0-100)
        comfort_score = 100
        if temp < 18 or temp > 26:
            comfort_score -= abs(temp - 22) * 5
        if humidity < 40 or humidity > 60:
            comfort_score -= abs(humidity - 50) * 0.5
        if wind_speed > 10:
            comfort_score -= (wind_speed - 10) * 2

        comfort_score = max(0, min(100, comfort_score))

        return {
            'heat_index': round(heat_index, 1),
            'comfort_score': round(comfort_score, 1),
            'thermal_stress': 'low' if comfort_score > 70 else 'moderate' if comfort_score > 40 else 'high'
        }

    def _calculate_energy_efficiency(self) -> Dict[str, float]:
        """Calculate energy efficiency metrics"""
        # Get energy consumption data
        consumption_values = []
        for sensor_data in self.sensor_history.values():
            for reading in sensor_data[-10:]:
                if reading.sensor_type == 'power_consumption':
                    consumption_values.append(reading.value)

        if not consumption_values:
            return {'efficiency_score': 75.0, 'consumption_trend': 'stable'}

        avg_consumption = statistics.mean(consumption_values)

        # Calculate efficiency score (lower consumption = higher efficiency)
        efficiency_score = max(0, 100 - (avg_consumption / 5.0) * 20)

        # Determine trend
        if len(consumption_values) >= 5:
            recent = statistics.mean(consumption_values[-3:])
            older = statistics.mean(consumption_values[:3])
            if recent > older * 1.1:
                trend = 'increasing'
            elif recent < older * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        return {
            'efficiency_score': round(efficiency_score, 1),
            'consumption_trend': trend,
            'avg_consumption': round(avg_consumption, 2)
        }

    def _calculate_water_metrics(self) -> Dict[str, float]:
        """Calculate water quality and usage metrics"""
        return {
            'quality_index': 85.0 + (hash(f"water_{time.time()}") % 20) - 10,
            'usage_efficiency': 78.0 + (hash(f"water_eff_{time.time()}") % 15) - 7,
            'contamination_risk': 'low'
        }

    def _calculate_biodiversity_indicators(self) -> Dict[str, float]:
        """Calculate biodiversity and ecosystem health"""
        return {
            'species_diversity': 72.0 + (hash(f"bio_{time.time()}") % 20) - 10,
            'vegetation_health': 80.0 + (hash(f"veg_{time.time()}") % 15) - 7,
            'habitat_connectivity': 65.0 + (hash(f"habitat_{time.time()}") % 25) - 12
        }

    def _calculate_urban_heat_island(self) -> float:
        """Calculate urban heat island intensity"""
        temp = self.weather_data.temperature
        # Simulate UHI effect (urban areas typically 2-5°C warmer)
        uhi_intensity = 2.5 + (hash(f"uhi_{time.time()}") % 25) / 10
        return round(uhi_intensity, 1)

    def _calculate_carbon_metrics(self) -> Dict[str, float]:
        """Calculate carbon footprint and emissions"""
        return {
            'co2_equivalent': 420.0 + (hash(f"co2_{time.time()}") % 50) - 25,
            'carbon_intensity': 0.8 + (hash(f"carbon_{time.time()}") % 40) / 100 - 0.2,
            'sequestration_potential': 15.0 + (hash(f"seq_{time.time()}") % 20) - 10
        }

    async def _generate_climate_recommendations(self):
        """Generate climate-responsive design recommendations"""
        recommendations = []
        current_time = datetime.now()

        # Always generate default recommendations first
        recommendations.extend(self._generate_default_recommendations())

        if not self.environmental_metrics or not self.weather_data:
            self.recommendations = recommendations
            return

        # Temperature-based recommendations
        temp = self.weather_data.temperature
        if temp > 30:
            recommendations.append(ClimateRecommendation(
                recommendation_id=f"heat_mitigation_{int(time.time())}",
                category="thermal_comfort",
                title="Implement Heat Mitigation Strategies",
                description="Deploy cooling solutions including green roofs, shade structures, and misting systems to combat high temperatures.",
                priority="high",
                impact_score=0.85,
                implementation_timeframe="1-3 months",
                cost_estimate="$50,000-150,000",
                environmental_triggers=[f"temperature_{temp}°C"],
                design_principles=["passive_cooling", "natural_ventilation", "thermal_mass"],
                technical_specifications={
                    "green_roof_coverage": "40%",
                    "shade_factor": "0.7",
                    "albedo_improvement": "0.3"
                }
            ))

        # Air quality recommendations
        aqi = self.environmental_metrics.air_quality['overall_aqi']
        if aqi > 100:
            recommendations.append(ClimateRecommendation(
                recommendation_id=f"air_quality_{int(time.time())}",
                category="air_quality",
                title="Air Purification Infrastructure",
                description="Install air purification systems and increase vegetation to improve air quality.",
                priority="high",
                impact_score=0.75,
                implementation_timeframe="2-6 months",
                cost_estimate="$100,000-300,000",
                environmental_triggers=[f"aqi_{aqi}"],
                design_principles=["natural_filtration", "pollution_reduction"],
                technical_specifications={
                    "tree_density": "150 trees/hectare",
                    "air_purifier_capacity": "1000 m³/hour"
                }
            ))

        # Energy efficiency recommendations
        energy_score = self.environmental_metrics.energy_efficiency['efficiency_score']
        if energy_score < 70:
            recommendations.append(ClimateRecommendation(
                recommendation_id=f"energy_efficiency_{int(time.time())}",
                category="energy",
                title="Renewable Energy Integration",
                description="Implement solar panels, energy storage, and smart grid systems to improve energy efficiency.",
                priority="medium",
                impact_score=0.80,
                implementation_timeframe="3-12 months",
                cost_estimate="$200,000-500,000",
                environmental_triggers=[f"energy_efficiency_{energy_score}%"],
                design_principles=["renewable_energy", "energy_storage", "smart_systems"],
                technical_specifications={
                    "solar_capacity": "500 kW",
                    "battery_storage": "2 MWh",
                    "efficiency_target": "85%"
                }
            ))

        # Precipitation-based recommendations
        if self.weather_data.precipitation > 5:
            recommendations.append(ClimateRecommendation(
                recommendation_id=f"stormwater_{int(time.time())}",
                category="water_management",
                title="Stormwater Management System",
                description="Implement rain gardens, permeable surfaces, and retention systems for flood prevention.",
                priority="medium",
                impact_score=0.70,
                implementation_timeframe="2-4 months",
                cost_estimate="$75,000-200,000",
                environmental_triggers=[f"precipitation_{self.weather_data.precipitation}mm"],
                design_principles=["natural_drainage", "water_retention", "flood_resilience"],
                technical_specifications={
                    "retention_capacity": "500,000 L",
                    "permeable_surface_ratio": "0.6",
                    "infiltration_rate": "25 mm/hour"
                }
            ))

        self.recommendations = recommendations

    def _generate_default_recommendations(self) -> List[ClimateRecommendation]:
        """Generate default recommendations when no metrics are available"""
        current_time = datetime.now()
        return [
            ClimateRecommendation(
                recommendation_id=f"default_energy_{int(time.time())}",
                category="energy",
                title="Implement Solar Energy Systems",
                description="Install photovoltaic panels and energy storage systems to improve renewable energy adoption and reduce carbon footprint.",
                priority="high",
                impact_score=0.85,
                implementation_timeframe="3-6 months",
                cost_estimate="$200,000-500,000",
                environmental_triggers=["baseline_assessment"],
                design_principles=["renewable_energy", "energy_storage", "grid_integration"],
                technical_specifications={
                    "solar_capacity": "1 MW",
                    "battery_storage": "5 MWh",
                    "efficiency_target": "90%"
                }
            ),
            ClimateRecommendation(
                recommendation_id=f"default_green_{int(time.time())}",
                category="biodiversity",
                title="Create Urban Green Corridors",
                description="Establish connected green spaces with native vegetation to enhance biodiversity and improve air quality.",
                priority="medium",
                impact_score=0.75,
                implementation_timeframe="6-12 months",
                cost_estimate="$150,000-400,000",
                environmental_triggers=["urban_planning"],
                design_principles=["native_species", "habitat_connectivity", "ecosystem_services"],
                technical_specifications={
                    "corridor_length": "2 km",
                    "vegetation_coverage": "80%",
                    "species_diversity": "50+ native species"
                }
            ),
            ClimateRecommendation(
                recommendation_id=f"default_water_{int(time.time())}",
                category="water_management",
                title="Smart Water Management System",
                description="Deploy IoT sensors and smart irrigation systems to optimize water usage and reduce waste.",
                priority="medium",
                impact_score=0.70,
                implementation_timeframe="2-4 months",
                cost_estimate="$75,000-200,000",
                environmental_triggers=["water_conservation"],
                design_principles=["smart_irrigation", "water_recycling", "leak_detection"],
                technical_specifications={
                    "sensor_network": "100 nodes",
                    "water_savings": "30%",
                    "monitoring_coverage": "95%"
                }
            )
        ]

    async def _check_environmental_alerts(self):
        """Check for environmental alerts and warnings"""
        if not self.environmental_metrics:
            return

        alerts = []

        # Temperature alerts
        temp = self.weather_data.temperature
        if temp > self.alert_thresholds['temperature']['heat_warning']:
            alerts.append({
                'type': 'heat_warning',
                'severity': 'high',
                'message': f"Extreme heat warning: {temp}°C",
                'recommendations': ['Seek air-conditioned spaces', 'Increase hydration']
            })

        # Air quality alerts
        aqi = self.environmental_metrics.air_quality['overall_aqi']
        if aqi > 150:
            alerts.append({
                'type': 'air_quality_alert',
                'severity': 'high',
                'message': f"Unhealthy air quality: AQI {aqi}",
                'recommendations': ['Limit outdoor activities', 'Use air purifiers indoors']
            })

        if alerts:
            logger.warning(f"⚠️ Environmental alerts: {len(alerts)} active")
            for alert in alerts:
                logger.warning(f"🚨 {alert['type']}: {alert['message']}")

    async def get_current_metrics(self, city='new-york') -> Optional[EnvironmentalMetrics]:
        """Get current environmental metrics for specific city"""
        # Always generate fresh metrics for any city including New York
        return await self._generate_city_metrics(city)

    async def get_sensor_data(self, city='new-york') -> Dict[str, List[SensorReading]]:
        """Get current sensor data for specific city"""
        if city != 'new-york':
            return await self._generate_city_sensors(city)
        return self.sensor_history

    async def get_weather_data(self, city='new-york') -> Optional[WeatherData]:
        """Get current weather data for specific city"""
        if city != 'new-york':
            return await self._generate_city_weather(city)
        # Generate fresh data for New York as well
        return await self._generate_city_weather('new-york')

    async def get_recommendations(self, city='new-york') -> List[ClimateRecommendation]:
        """Get current climate-responsive recommendations for specific city"""
        if city != 'new-york':
            return await self._generate_city_recommendations(city)
        return self.recommendations

    async def get_iot_sensors(self, city='new-york') -> Dict[str, Any]:
        """Get IoT sensor data for specific city"""
        # Always generate fresh IoT sensor data for any city
        return await self._generate_city_iot_sensors(city)

    async def get_historical_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical environmental data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter historical data
        historical_sensors = {}
        for sensor_id, readings in self.sensor_history.items():
            historical_sensors[sensor_id] = [
                r for r in readings if r.timestamp > cutoff_time
            ]

        historical_weather = [
            w for w in self.weather_history if w.timestamp > cutoff_time
        ]

        return {
            'sensor_data': historical_sensors,
            'weather_data': historical_weather,
            'time_range': f"{hours} hours"
        }

    async def _generate_city_metrics(self, city: str) -> EnvironmentalMetrics:
        """Generate environmental metrics for specific city"""
        import random

        # Comprehensive city configurations with real-time data
        city_configs = {
            'new-york': {'base_aqi': 75, 'pollution': 0.65, 'green_space': 0.27, 'temp': 12, 'humidity': 68, 'coords': {'lat': 40.7589, 'lon': -73.9851}},
            'london': {'base_aqi': 60, 'pollution': 0.55, 'green_space': 0.33, 'temp': 8, 'humidity': 78, 'coords': {'lat': 51.5074, 'lon': -0.1278}},
            'tokyo': {'base_aqi': 45, 'pollution': 0.45, 'green_space': 0.25, 'temp': 16, 'humidity': 65, 'coords': {'lat': 35.6762, 'lon': 139.6503}},
            'singapore': {'base_aqi': 35, 'pollution': 0.35, 'green_space': 0.47, 'temp': 28, 'humidity': 84, 'coords': {'lat': 1.3521, 'lon': 103.8198}},
            'sydney': {'base_aqi': 25, 'pollution': 0.25, 'green_space': 0.46, 'temp': 20, 'humidity': 65, 'coords': {'lat': -33.8688, 'lon': 151.2093}},
            'copenhagen': {'base_aqi': 20, 'pollution': 0.20, 'green_space': 0.56, 'temp': 6, 'humidity': 75, 'coords': {'lat': 55.6761, 'lon': 12.5683}},
            'vancouver': {'base_aqi': 15, 'pollution': 0.18, 'green_space': 0.54, 'temp': 10, 'humidity': 72, 'coords': {'lat': 49.2827, 'lon': -123.1207}},
            'amsterdam': {'base_aqi': 30, 'pollution': 0.30, 'green_space': 0.45, 'temp': 9, 'humidity': 76, 'coords': {'lat': 52.3676, 'lon': 4.9041}},
            'berlin': {'base_aqi': 35, 'pollution': 0.35, 'green_space': 0.44, 'temp': 7, 'humidity': 73, 'coords': {'lat': 52.5200, 'lon': 13.4050}},
            'stockholm': {'base_aqi': 18, 'pollution': 0.15, 'green_space': 0.58, 'temp': 4, 'humidity': 74, 'coords': {'lat': 59.3293, 'lon': 18.0686}},
            # Indian Cities - High pollution, moderate green space
            'mumbai': {'base_aqi': 145, 'pollution': 0.78, 'green_space': 0.16, 'temp': 27, 'humidity': 75, 'coords': {'lat': 19.0760, 'lon': 72.8777}},
            'delhi': {'base_aqi': 165, 'pollution': 0.85, 'green_space': 0.22, 'temp': 25, 'humidity': 65, 'coords': {'lat': 28.7041, 'lon': 77.1025}},
            'bangalore': {'base_aqi': 95, 'pollution': 0.58, 'green_space': 0.31, 'temp': 24, 'humidity': 65, 'coords': {'lat': 12.9716, 'lon': 77.5946}},
            # Middle East & Asia
            'dubai': {'base_aqi': 85, 'pollution': 0.48, 'green_space': 0.12, 'temp': 32, 'humidity': 55, 'coords': {'lat': 25.2048, 'lon': 55.2708}},
            'seoul': {'base_aqi': 55, 'pollution': 0.42, 'green_space': 0.28, 'temp': 12, 'humidity': 68, 'coords': {'lat': 37.5665, 'lon': 126.9780}},
            'hong-kong': {'base_aqi': 65, 'pollution': 0.45, 'green_space': 0.24, 'temp': 23, 'humidity': 78, 'coords': {'lat': 22.3193, 'lon': 114.1694}},
            # Clean European Cities
            'zurich': {'base_aqi': 12, 'pollution': 0.12, 'green_space': 0.62, 'temp': 9, 'humidity': 71, 'coords': {'lat': 47.3769, 'lon': 8.5417}},
            'oslo': {'base_aqi': 8, 'pollution': 0.08, 'green_space': 0.68, 'temp': 5, 'humidity': 73, 'coords': {'lat': 59.9139, 'lon': 10.7522}},
            'reykjavik': {'base_aqi': 5, 'pollution': 0.05, 'green_space': 0.75, 'temp': 4, 'humidity': 76, 'coords': {'lat': 64.1466, 'lon': -21.9426}},
            # Other Continents
            'cape-town': {'base_aqi': 28, 'pollution': 0.22, 'green_space': 0.52, 'temp': 18, 'humidity': 68, 'coords': {'lat': -33.9249, 'lon': 18.4241}},
            'sao-paulo': {'base_aqi': 95, 'pollution': 0.68, 'green_space': 0.18, 'temp': 20, 'humidity': 70, 'coords': {'lat': -23.5505, 'lon': -46.6333}},
            'mexico-city': {'base_aqi': 125, 'pollution': 0.72, 'green_space': 0.15, 'temp': 18, 'humidity': 55, 'coords': {'lat': 19.4326, 'lon': -99.1332}},
            'cairo': {'base_aqi': 135, 'pollution': 0.75, 'green_space': 0.08, 'temp': 23, 'humidity': 45, 'coords': {'lat': 30.0444, 'lon': 31.2357}},
            'istanbul': {'base_aqi': 85, 'pollution': 0.55, 'green_space': 0.20, 'temp': 14, 'humidity': 68, 'coords': {'lat': 41.0082, 'lon': 28.9784}},
            'moscow': {'base_aqi': 48, 'pollution': 0.38, 'green_space': 0.35, 'temp': 5, 'humidity': 70, 'coords': {'lat': 55.7558, 'lon': 37.6176}}
        }

        config = city_configs.get(city, city_configs['london'])

        # Generate sophisticated time-based variations
        import math
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second
        weekday = now.weekday()

        # Daily pollution cycle (rush hours, industrial activity)
        rush_hour_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Morning/evening rush
            rush_hour_factor = 1.4 + 0.1 * math.sin(minute * math.pi / 30)
        elif 10 <= hour <= 16:  # Daytime activity
            rush_hour_factor = 1.2 + 0.1 * math.cos(hour * math.pi / 12)
        elif 22 <= hour or hour <= 5:  # Night hours
            rush_hour_factor = 0.6 + 0.1 * math.sin(hour * math.pi / 6)

        # Weekend vs weekday patterns
        weekend_factor = 0.75 if weekday >= 5 else 1.0

        # Micro-variations based on seconds for real-time feel
        micro_variation = 1.0 + 0.02 * math.sin(second * math.pi / 30)

        # Seasonal factor based on day of year
        day_of_year = now.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.25 * math.cos((day_of_year - 15) * 2 * math.pi / 365)

        # Combined dynamic factor
        dynamic_factor = rush_hour_factor * weekend_factor * micro_variation * seasonal_factor

        # AQI calculation with sophisticated dynamic variations
        base_pollution = config['base_aqi'] * dynamic_factor
        current_aqi = max(5, base_pollution + random.uniform(-12, 12))

        # PM2.5 and NO2 with realistic correlations
        pm25 = max(1, current_aqi * 0.65 + random.uniform(-6, 8) * rush_hour_factor)
        no2 = max(2, current_aqi * 0.45 + random.uniform(-4, 6) * rush_hour_factor)

        # CO varies more with traffic patterns
        co_level = max(0.1, config['pollution'] * 18 * rush_hour_factor + random.uniform(-4, 4))

        # O3 has different patterns (higher in afternoon heat)
        ozone_factor = 1.3 if 12 <= hour <= 16 else 0.8
        o3_level = max(5, config['pollution'] * 30 * ozone_factor + random.uniform(-8, 8))

        if current_aqi <= 50:
            health_risk = 'Good'
        elif current_aqi <= 100:
            health_risk = 'Moderate'
        elif current_aqi <= 150:
            health_risk = 'Unhealthy for Sensitive Groups'
        elif current_aqi <= 200:
            health_risk = 'Unhealthy'
        else:
            health_risk = 'Very Unhealthy'

        return EnvironmentalMetrics(
            timestamp=datetime.now(),
            location=config['coords'],
            air_quality={
                'overall_aqi': round(current_aqi, 1),
                'pm25_level': round(pm25, 1),
                'no2_level': round(no2, 1),
                'health_risk': health_risk,
                'co_level': round(co_level, 1),
                'o3_level': round(o3_level, 1)
            },
            climate_comfort={
                'comfort_score': round(max(0, min(100, 92 - config['pollution'] * 35 * dynamic_factor + random.uniform(-10, 10))), 1),
                'temperature_comfort': round(9.2 - config['pollution'] * 2.5 * seasonal_factor + random.uniform(-1.5, 1.5), 1),
                'humidity_optimal': config['humidity'] > 35 and config['humidity'] < 85,
                'thermal_index': round(config['temp'] * seasonal_factor + config['humidity'] * 0.12 + random.uniform(-2.5, 2.5), 1)
            },
            energy_efficiency={
                'efficiency_score': round(max(0, min(100, 78 + config['green_space'] * 28 * weekend_factor + random.uniform(-12, 12))), 1),
                'renewable_percentage': round(max(0, min(100, 52 + config['green_space'] * 55 + random.uniform(-15, 15))), 1),
                'grid_stability': round(87 + config['green_space'] * 18 - rush_hour_factor * 5 + random.uniform(-7, 7), 1),
                'peak_demand': round(config['pollution'] * 125 * rush_hour_factor + 65 + random.uniform(-20, 20), 1)
            },
            water_metrics={
                'quality_index': round(max(0, min(100, 91 + config['green_space'] * 15 - config['pollution'] * 18 + random.uniform(-8, 8))), 1),
                'usage_efficiency': round(max(0, min(100, 68 + config['green_space'] * 35 * weekend_factor + random.uniform(-12, 12))), 1),
                'treatment_effectiveness': round(max(0, min(100, 94 - config['pollution'] * 12 + random.uniform(-6, 6))), 1),
                'conservation_rate': round(config['green_space'] * 85 + 22 + random.uniform(-10, 10), 1)
            },
            biodiversity_indicators={
                'species_count': max(5, int(config['green_space'] * 280 * seasonal_factor + random.randint(-35, 35))),
                'habitat_quality': round(max(0, min(100, config['green_space'] * 98 * seasonal_factor + random.uniform(-15, 15))), 1),
                'ecosystem_connectivity': round(config['green_space'] * 95 + 12 + random.uniform(-10, 10), 1),
                'native_species_ratio': round(config['green_space'] * 75 + 32 + random.uniform(-12, 12), 1)
            },
            urban_heat_island=round(config['pollution'] * 3.5 * dynamic_factor + config['temp'] * 0.12 + random.uniform(-1.2, 1.2), 2),
            carbon_metrics={
                'carbon_intensity': round(config['pollution'] * 1.4 * dynamic_factor + random.uniform(-0.2, 0.2), 3),
                'sequestration_rate': round(config['green_space'] * 12 + random.uniform(-2.5, 2.5), 2),
                'emissions_per_capita': round(config['pollution'] * 8 + 2 + random.uniform(-1, 1), 2),
                'renewable_adoption': round(config['green_space'] * 60 + 25 + random.uniform(-8, 8), 1)
            }
        )

    async def _generate_city_weather(self, city: str) -> WeatherData:
        """Generate comprehensive weather data for specific city"""
        import random
        import math

        # Use the comprehensive city configs from above
        configs = {
            'mumbai': {'temp': 27, 'humidity': 75, 'pressure': 1010, 'wind': 8, 'uv': 9},
            'delhi': {'temp': 25, 'humidity': 65, 'pressure': 1012, 'wind': 6, 'uv': 8},
            'bangalore': {'temp': 24, 'humidity': 65, 'pressure': 1015, 'wind': 4, 'uv': 7},
            'dubai': {'temp': 32, 'humidity': 55, 'pressure': 1008, 'wind': 12, 'uv': 10},
            'seoul': {'temp': 12, 'humidity': 68, 'pressure': 1018, 'wind': 7, 'uv': 4},
            'hong-kong': {'temp': 23, 'humidity': 78, 'pressure': 1014, 'wind': 9, 'uv': 6},
            'zurich': {'temp': 9, 'humidity': 71, 'pressure': 1020, 'wind': 5, 'uv': 3},
            'oslo': {'temp': 5, 'humidity': 73, 'pressure': 1022, 'wind': 6, 'uv': 2},
            'reykjavik': {'temp': 4, 'humidity': 76, 'pressure': 1018, 'wind': 15, 'uv': 1},
            'cape-town': {'temp': 18, 'humidity': 68, 'pressure': 1016, 'wind': 12, 'uv': 8},
            'sao-paulo': {'temp': 20, 'humidity': 70, 'pressure': 1012, 'wind': 5, 'uv': 6},
            'mexico-city': {'temp': 18, 'humidity': 55, 'pressure': 1009, 'wind': 4, 'uv': 7},
            'cairo': {'temp': 23, 'humidity': 45, 'pressure': 1013, 'wind': 8, 'uv': 9},
            'istanbul': {'temp': 14, 'humidity': 68, 'pressure': 1015, 'wind': 7, 'uv': 5},
            'moscow': {'temp': 5, 'humidity': 70, 'pressure': 1019, 'wind': 6, 'uv': 2},
            'london': {'temp': 8, 'humidity': 78, 'pressure': 1013, 'wind': 9, 'uv': 3},
            'tokyo': {'temp': 16, 'humidity': 65, 'pressure': 1016, 'wind': 6, 'uv': 5},
            'singapore': {'temp': 28, 'humidity': 84, 'pressure': 1011, 'wind': 4, 'uv': 9},
            'sydney': {'temp': 20, 'humidity': 65, 'pressure': 1017, 'wind': 11, 'uv': 7},
            'copenhagen': {'temp': 6, 'humidity': 75, 'pressure': 1019, 'wind': 8, 'uv': 2},
            'vancouver': {'temp': 10, 'humidity': 72, 'pressure': 1014, 'wind': 7, 'uv': 3},
            'amsterdam': {'temp': 9, 'humidity': 76, 'pressure': 1016, 'wind': 9, 'uv': 3},
            'berlin': {'temp': 7, 'humidity': 73, 'pressure': 1017, 'wind': 6, 'uv': 3},
            'stockholm': {'temp': 4, 'humidity': 74, 'pressure': 1020, 'wind': 7, 'uv': 2},
            'new-york': {'temp': 12, 'humidity': 68, 'pressure': 1015, 'wind': 9, 'uv': 4}
        }

        config = configs.get(city, configs['new-york'])

        # Add seasonal and daily variations
        hour = datetime.now().hour
        seasonal_factor = 0.8 + 0.4 * math.sin(2 * math.pi * hour / 24)

        return WeatherData(
            location=city.replace('-', ' ').title(),
            timestamp=datetime.now(),
            temperature=round(config['temp'] + random.uniform(-4, 4) * seasonal_factor, 1),
            humidity=round(max(20, min(95, config['humidity'] + random.uniform(-12, 12))), 1),
            pressure=round(config['pressure'] + random.uniform(-8, 8), 1),
            wind_speed=round(max(0, config['wind'] + random.uniform(-3, 3)), 1),
            wind_direction=round(random.uniform(0, 360), 0),
            precipitation=round(max(0, random.uniform(0, 3) if config['humidity'] > 70 else random.uniform(0, 1)), 1),
            cloud_cover=round(max(0, min(100, config['humidity'] * 0.8 + random.uniform(-20, 20))), 0),
            visibility=round(max(1, min(20, 15 - (config['humidity'] - 50) * 0.1 + random.uniform(-2, 2))), 1),
            uv_index=round(max(0, min(11, config['uv'] + random.uniform(-1, 1) * seasonal_factor)), 1),
            air_quality_index=None
        )

    async def _generate_city_recommendations(self, city: str) -> List[ClimateRecommendation]:
        """Generate dynamic, weather and climate-responsive recommendations"""
        import random
        import math

        # Get current weather and environmental data for the city
        weather_configs = {
            'mumbai': {'temp': 27, 'humidity': 75, 'aqi': 145, 'wind': 8, 'climate': 'tropical_monsoon'},
            'delhi': {'temp': 25, 'humidity': 65, 'aqi': 165, 'wind': 6, 'climate': 'hot_semi_arid'},
            'bangalore': {'temp': 24, 'humidity': 65, 'aqi': 95, 'wind': 4, 'climate': 'tropical_savanna'},
            'dubai': {'temp': 32, 'humidity': 55, 'aqi': 85, 'wind': 12, 'climate': 'hot_desert'},
            'reykjavik': {'temp': 4, 'humidity': 76, 'aqi': 5, 'wind': 15, 'climate': 'subarctic'},
            'tokyo': {'temp': 16, 'humidity': 65, 'aqi': 45, 'wind': 6, 'climate': 'humid_subtropical'},
            'new-york': {'temp': 12, 'humidity': 68, 'aqi': 75, 'wind': 9, 'climate': 'humid_continental'},
            'london': {'temp': 8, 'humidity': 78, 'aqi': 60, 'wind': 9, 'climate': 'temperate_oceanic'},
            'singapore': {'temp': 28, 'humidity': 84, 'aqi': 35, 'wind': 4, 'climate': 'tropical_rainforest'},
            'sydney': {'temp': 20, 'humidity': 65, 'aqi': 25, 'wind': 11, 'climate': 'humid_subtropical'}
        }

        config = weather_configs.get(city, weather_configs['new-york'])
        city_name = city.replace('-', ' ').title()
        now = datetime.now()
        hour = now.hour

        # Add real-time weather variations
        current_temp = config['temp'] + random.uniform(-3, 3) + 2 * math.sin(hour * math.pi / 12)
        current_humidity = max(20, min(95, config['humidity'] + random.uniform(-10, 10)))
        current_aqi = max(5, config['aqi'] + random.uniform(-15, 15))
        current_wind = max(0, config['wind'] + random.uniform(-3, 3))

        recommendations = []

        # Weather-responsive primary recommendation
        if current_temp > 30 and config['climate'] in ['hot_desert', 'tropical_monsoon']:
            # Hot climate cooling solutions
            cooling_impact = round(88 + random.uniform(-5, 8), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'cooling_{city}_{hour}_{now.minute}',
                category='ENERGY',
                priority='HIGH',
                title=f'Advanced Cooling Systems for {city_name}',
                description=f'Deploy energy-efficient cooling solutions optimized for {current_temp:.1f}°C temperatures and {current_humidity:.0f}% humidity levels.',
                impact_score=cooling_impact,
                implementation_timeframe='2-4 months',
                cost_estimate='$250,000-600,000',
                environmental_triggers=[f'High temperature: {current_temp:.1f}°C', f'Humidity: {current_humidity:.0f}%', 'Energy efficiency'],
                design_principles=['passive cooling', 'thermal mass', 'natural ventilation'],
                technical_specifications={'cooling_capacity': f'{int(current_temp*100)}kW', 'efficiency': '95%', 'temp_reduction': f'{current_temp-22:.1f}°C'}
            ))
        elif current_temp < 10 and config['climate'] in ['subarctic', 'humid_continental']:
            # Cold climate heating solutions
            heating_impact = round(85 + random.uniform(-6, 10), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'heating_{city}_{hour}_{now.minute}',
                category='ENERGY',
                priority='HIGH',
                title=f'Efficient Heating Systems for {city_name}',
                description=f'Install renewable heating solutions for {current_temp:.1f}°C conditions with {current_wind:.1f} m/s wind exposure.',
                impact_score=heating_impact,
                implementation_timeframe='3-6 months',
                cost_estimate='$300,000-750,000',
                environmental_triggers=[f'Low temperature: {current_temp:.1f}°C', f'Wind chill: {current_wind:.1f} m/s', 'Heating demand'],
                design_principles=['geothermal energy', 'heat pumps', 'insulation'],
                technical_specifications={'heating_capacity': f'{int(abs(current_temp-20)*50)}kW', 'cop': '4.5', 'temp_gain': f'{20-current_temp:.1f}°C'}
            ))
        elif current_aqi > 100:
            # High pollution air quality solutions
            air_impact = round(92 + random.uniform(-4, 6), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'air_quality_{city}_{hour}_{now.minute}',
                category='ENERGY',
                priority='HIGH',
                title=f'Emergency Air Purification for {city_name}',
                description=f'Deploy advanced air filtration systems to combat AQI {current_aqi:.0f} pollution levels with {current_wind:.1f} m/s wind dispersion.',
                impact_score=air_impact,
                implementation_timeframe='1-3 months',
                cost_estimate='$400,000-900,000',
                environmental_triggers=[f'Critical AQI: {current_aqi:.0f}', f'Wind dispersion: {current_wind:.1f} m/s', 'Health emergency'],
                design_principles=['HEPA filtration', 'green walls', 'air circulation'],
                technical_specifications={'purification_rate': f'{int(current_aqi*50)} m³/h', 'efficiency': '99.97%', 'aqi_reduction': f'{current_aqi-50:.0f}'}
            ))
        else:
            # Balanced renewable energy solutions
            energy_impact = round(80 + random.uniform(-8, 12), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'renewable_{city}_{hour}_{now.minute}',
                category='ENERGY',
                priority='HIGH',
                title=f'Climate-Optimized Renewable Energy for {city_name}',
                description=f'Install weather-adaptive renewable systems for {current_temp:.1f}°C, {current_humidity:.0f}% humidity, and {current_wind:.1f} m/s wind conditions.',
                impact_score=energy_impact,
                implementation_timeframe='4-8 months',
                cost_estimate='$200,000-500,000',
                environmental_triggers=[f'Temperature: {current_temp:.1f}°C', f'Wind: {current_wind:.1f} m/s', 'Renewable potential'],
                design_principles=['solar optimization', 'wind integration', 'smart grid'],
                technical_specifications={'solar_capacity': f'{max(1, int(current_temp/10))}MW', 'wind_capacity': f'{max(0.5, current_wind/10):.1f}MW', 'efficiency': f'{85+int(current_wind)}%'}
            ))

        # Humidity-responsive water management
        if current_humidity > 75:
            # High humidity water solutions
            water_impact = round(75 + random.uniform(-8, 12), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'humidity_water_{city}_{now.minute}_{now.second}',
                category='WATER MANAGEMENT',
                priority='MEDIUM',
                title=f'Humidity Control Water Systems for {city_name}',
                description=f'Deploy dehumidification and water harvesting systems for {current_humidity:.0f}% humidity conditions.',
                impact_score=water_impact,
                implementation_timeframe='2-5 months',
                cost_estimate='$100,000-250,000',
                environmental_triggers=[f'High humidity: {current_humidity:.0f}%', 'Water harvesting potential', 'Moisture control'],
                design_principles=['atmospheric water generation', 'dehumidification', 'water recycling'],
                technical_specifications={'water_harvest': f'{int(current_humidity*0.5)}L/day', 'dehumidification': f'{current_humidity-60:.0f}%', 'efficiency': f'{min(95, 60+current_humidity/2):.0f}%'}
            ))
        else:
            # Low humidity water conservation
            water_impact = round(72 + random.uniform(-10, 15), 0)
            recommendations.append(ClimateRecommendation(
                recommendation_id=f'conservation_water_{city}_{now.minute}_{now.second}',
                category='WATER MANAGEMENT',
                priority='MEDIUM',
                title=f'Water Conservation Systems for {city_name}',
                description=f'Implement smart irrigation and conservation for {current_humidity:.0f}% humidity and {current_temp:.1f}°C conditions.',
                impact_score=water_impact,
                implementation_timeframe='2-4 months',
                cost_estimate='$75,000-200,000',
                environmental_triggers=[f'Low humidity: {current_humidity:.0f}%', f'Temperature: {current_temp:.1f}°C', 'Water scarcity risk'],
                design_principles=['smart irrigation', 'drought resistance', 'water efficiency'],
                technical_specifications={'water_savings': f'{max(20, 100-current_humidity):.0f}%', 'coverage': f'{int(current_temp*2)} km²', 'sensors': f'{int(current_temp*10)}'}
            ))

        # Climate-adaptive biodiversity solutions
        bio_impact = round(65 + random.uniform(-12, 18), 0)
        if config['climate'] in ['tropical_rainforest', 'tropical_monsoon']:
            bio_focus = 'Tropical Species Integration'
            bio_description = f'Establish climate-resilient tropical ecosystems adapted to {current_temp:.1f}°C and {current_humidity:.0f}% humidity.'
            bio_principles = ['tropical species', 'humidity adaptation', 'carbon sequestration']
        elif config['climate'] in ['hot_desert', 'hot_semi_arid']:
            bio_focus = 'Desert Ecosystem Restoration'
            bio_description = f'Create drought-resistant green spaces for {current_temp:.1f}°C desert conditions with {current_humidity:.0f}% humidity.'
            bio_principles = ['xerophytic plants', 'water conservation', 'heat resistance']
        elif config['climate'] in ['subarctic', 'humid_continental']:
            bio_focus = 'Cold-Climate Biodiversity'
            bio_description = f'Develop cold-adapted ecosystems for {current_temp:.1f}°C temperatures and {current_wind:.1f} m/s wind exposure.'
            bio_principles = ['cold-hardy species', 'wind protection', 'seasonal adaptation']
        else:
            bio_focus = 'Temperate Ecosystem Enhancement'
            bio_description = f'Create balanced temperate ecosystems for {current_temp:.1f}°C and {current_humidity:.0f}% humidity conditions.'
            bio_principles = ['native adaptation', 'seasonal variation', 'ecosystem balance']

        recommendations.append(ClimateRecommendation(
            recommendation_id=f'biodiversity_{city}_{now.hour}_{now.second}',
            category='BIODIVERSITY',
            priority='MEDIUM',
            title=bio_focus,
            description=bio_description,
            impact_score=bio_impact,
            implementation_timeframe='6-12 months',
            cost_estimate='$150,000-400,000',
            environmental_triggers=[f'Climate: {config["climate"]}', f'Temperature: {current_temp:.1f}°C', f'Humidity: {current_humidity:.0f}%'],
            design_principles=bio_principles,
            technical_specifications={'area': f'{max(5, int(current_temp))} hectares', 'species': f'{max(20, int(current_humidity*0.8))}+', 'adaptation_score': f'{bio_impact}%'}
        ))

        return recommendations

    async def _generate_city_sensors(self, city: str) -> Dict[str, List[SensorReading]]:
        """Generate sensor data for specific city"""
        return {'downtown': []}

    async def _generate_city_iot_sensors(self, city: str) -> Dict[str, Any]:
        """Generate comprehensive IoT sensor data for specific city"""
        import random
        import math

        # City-specific configurations
        city_configs = {
            'new-york': {'base_aqi': 75, 'pollution': 0.65, 'green_space': 0.27, 'temp': 12, 'humidity': 68, 'coords': {'lat': 40.7589, 'lon': -73.9851}},
            'london': {'base_aqi': 60, 'pollution': 0.55, 'green_space': 0.33, 'temp': 8, 'humidity': 78, 'coords': {'lat': 51.5074, 'lon': -0.1278}},
            'tokyo': {'base_aqi': 45, 'pollution': 0.45, 'green_space': 0.25, 'temp': 16, 'humidity': 65, 'coords': {'lat': 35.6762, 'lon': 139.6503}},
            'singapore': {'base_aqi': 35, 'pollution': 0.35, 'green_space': 0.47, 'temp': 28, 'humidity': 84, 'coords': {'lat': 1.3521, 'lon': 103.8198}},
            'sydney': {'base_aqi': 25, 'pollution': 0.25, 'green_space': 0.46, 'temp': 20, 'humidity': 65, 'coords': {'lat': -33.8688, 'lon': 151.2093}},
            'mumbai': {'base_aqi': 145, 'pollution': 0.78, 'green_space': 0.16, 'temp': 27, 'humidity': 75, 'coords': {'lat': 19.0760, 'lon': 72.8777}},
            'delhi': {'base_aqi': 165, 'pollution': 0.85, 'green_space': 0.22, 'temp': 25, 'humidity': 65, 'coords': {'lat': 28.7041, 'lon': 77.1025}},
            'bangalore': {'base_aqi': 95, 'pollution': 0.58, 'green_space': 0.31, 'temp': 24, 'humidity': 65, 'coords': {'lat': 12.9716, 'lon': 77.5946}},
            'dubai': {'base_aqi': 85, 'pollution': 0.48, 'green_space': 0.12, 'temp': 32, 'humidity': 55, 'coords': {'lat': 25.2048, 'lon': 55.2708}},
            'reykjavik': {'base_aqi': 5, 'pollution': 0.05, 'green_space': 0.75, 'temp': 4, 'humidity': 76, 'coords': {'lat': 64.1466, 'lon': -21.9426}}
        }

        config = city_configs.get(city, city_configs['new-york'])
        now = datetime.now()
        hour = now.hour

        # Time-based dynamic factors
        rush_hour_factor = 1.4 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0.8
        time_micro_variation = 1.0 + 0.05 * math.sin(now.second * math.pi / 30)

        # Generate comprehensive sensor data
        sensors_data = {}

        # Air Quality Downtown Sensor
        no2_base = config['base_aqi'] * 0.6 * rush_hour_factor * time_micro_variation
        sensors_data['air_quality_downtown'] = [{
            'sensor_id': 'air_quality_downtown',
            'sensor_type': 'no2',
            'location': config['coords'],
            'timestamp': now.isoformat(),
            'value': round(max(5, no2_base + random.uniform(-8, 12)), 2),
            'unit': 'μg/m³',
            'quality_score': round(max(0.6, min(0.98, 0.9 - config['pollution'] * 0.3 + random.uniform(-0.1, 0.1))), 2),
            'metadata': {'calibration_date': '2024-11-01'}
        }]

        # Weather Station
        humidity_value = config['humidity'] + random.uniform(-8, 8) * time_micro_variation
        sensors_data['weather_station_01'] = [{
            'sensor_id': 'weather_station_01',
            'sensor_type': 'humidity',
            'location': {'lat': config['coords']['lat'] + 0.0025, 'lon': config['coords']['lon'] + 0.0025},
            'timestamp': now.isoformat(),
            'value': round(max(20, min(95, humidity_value)), 2),
            'unit': '%',
            'quality_score': round(random.uniform(0.88, 0.98), 2),
            'metadata': {'sensor_model': 'DHT22'}
        }]

        # Energy Meter Residential
        base_consumption = 1.2 + config['pollution'] * 0.8 + rush_hour_factor * 0.6
        energy_consumption = base_consumption * time_micro_variation + random.uniform(-0.3, 0.3)
        sensors_data['energy_meter_residential'] = [{
            'sensor_id': 'energy_meter_residential',
            'sensor_type': 'power_consumption',
            'location': {'lat': config['coords']['lat'] - 0.005, 'lon': config['coords']['lon'] + 0.01},
            'timestamp': now.isoformat(),
            'value': round(max(0.5, energy_consumption), 2),
            'unit': 'kWh',
            'quality_score': round(random.uniform(0.95, 0.99), 2),
            'metadata': {'meter_type': 'smart_grid'}
        }]

        # Water Quality Sensor
        water_quality = max(60, 95 - config['pollution'] * 30 + random.uniform(-5, 8))
        sensors_data['water_quality_01'] = [{
            'sensor_id': 'water_quality_01',
            'sensor_type': 'ph_level',
            'location': {'lat': config['coords']['lat'] + 0.002, 'lon': config['coords']['lon'] - 0.003},
            'timestamp': now.isoformat(),
            'value': round(6.8 + random.uniform(-0.5, 0.8), 2),
            'unit': 'pH',
            'quality_score': round(water_quality / 100, 2),
            'metadata': {'sensor_depth': '2.5m'}
        }]

        # Noise Monitor
        noise_level = 45 + config['pollution'] * 25 + rush_hour_factor * 15 + random.uniform(-5, 10)
        sensors_data['noise_monitor_commercial'] = [{
            'sensor_id': 'noise_monitor_commercial',
            'sensor_type': 'decibel_level',
            'location': {'lat': config['coords']['lat'] - 0.001, 'lon': config['coords']['lon'] + 0.002},
            'timestamp': now.isoformat(),
            'value': round(max(35, min(85, noise_level)), 1),
            'unit': 'dB',
            'quality_score': round(random.uniform(0.85, 0.95), 2),
            'metadata': {'frequency_range': '20Hz-20kHz'}
        }]

        return sensors_data

    async def shutdown(self):
        """Shutdown the environmental system"""
        logger.info("🛑 Shutting down environmental monitoring system...")
        self.running = False