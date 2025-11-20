#!/usr/bin/env python3
"""
RegeneraX Server - Web server with API endpoints for the dashboard
Provides both the demo functionality and web API endpoints
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta
import random
import statistics
from typing import Dict, List, Any, Optional
from pathlib import Path
from core.realtime_environmental_system import RealTimeEnvironmentalSystem
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegeneraXServer:
    """Web server that provides API endpoints for the dashboard"""

    def __init__(self, port=8000):
        self.port = port
        self.sensors = {}
        self.city_data = {}
        self.running = False
        self.start_time = None
        self.environmental_system = RealTimeEnvironmentalSystem()
        self.initialize_data()

    def initialize_data(self):
        """Initialize demo data"""
        # City-specific configurations
        self.city_configs = {
            'new-york': {
                'name': 'New York City',
                'coordinates': {'lat': 40.7589, 'lon': -73.9851},
                'timezone': 'America/New_York',
                'base_temp': 12, 'temp_range': 15,
                'base_aqi': 75, 'aqi_range': 30,
                'base_humidity': 68, 'humidity_range': 25,
                'pollution_level': 0.65, 'green_space': 0.27
            },
            'london': {
                'name': 'London',
                'coordinates': {'lat': 51.5074, 'lon': -0.1278},
                'timezone': 'Europe/London',
                'base_temp': 8, 'temp_range': 12,
                'base_aqi': 60, 'aqi_range': 25,
                'base_humidity': 78, 'humidity_range': 20,
                'pollution_level': 0.55, 'green_space': 0.33
            },
            'tokyo': {
                'name': 'Tokyo',
                'coordinates': {'lat': 35.6762, 'lon': 139.6503},
                'timezone': 'Asia/Tokyo',
                'base_temp': 16, 'temp_range': 18,
                'base_aqi': 45, 'aqi_range': 20,
                'base_humidity': 65, 'humidity_range': 30,
                'pollution_level': 0.45, 'green_space': 0.25
            },
            'singapore': {
                'name': 'Singapore',
                'coordinates': {'lat': 1.3521, 'lon': 103.8198},
                'timezone': 'Asia/Singapore',
                'base_temp': 28, 'temp_range': 6,
                'base_aqi': 35, 'aqi_range': 25,
                'base_humidity': 84, 'humidity_range': 15,
                'pollution_level': 0.35, 'green_space': 0.47
            },
            'sydney': {
                'name': 'Sydney',
                'coordinates': {'lat': -33.8688, 'lon': 151.2093},
                'timezone': 'Australia/Sydney',
                'base_temp': 20, 'temp_range': 12,
                'base_aqi': 25, 'aqi_range': 15,
                'base_humidity': 65, 'humidity_range': 25,
                'pollution_level': 0.25, 'green_space': 0.46
            },
            'copenhagen': {
                'name': 'Copenhagen',
                'coordinates': {'lat': 55.6761, 'lon': 12.5683},
                'timezone': 'Europe/Copenhagen',
                'base_temp': 6, 'temp_range': 10,
                'base_aqi': 20, 'aqi_range': 15,
                'base_humidity': 75, 'humidity_range': 20,
                'pollution_level': 0.20, 'green_space': 0.56
            },
            'vancouver': {
                'name': 'Vancouver',
                'coordinates': {'lat': 49.2827, 'lon': -123.1207},
                'timezone': 'America/Vancouver',
                'base_temp': 10, 'temp_range': 12,
                'base_aqi': 15, 'aqi_range': 20,
                'base_humidity': 72, 'humidity_range': 25,
                'pollution_level': 0.18, 'green_space': 0.54
            },
            'amsterdam': {
                'name': 'Amsterdam',
                'coordinates': {'lat': 52.3676, 'lon': 4.9041},
                'timezone': 'Europe/Amsterdam',
                'base_temp': 9, 'temp_range': 11,
                'base_aqi': 30, 'aqi_range': 20,
                'base_humidity': 76, 'humidity_range': 20,
                'pollution_level': 0.30, 'green_space': 0.45
            },
            'berlin': {
                'name': 'Berlin',
                'coordinates': {'lat': 52.5200, 'lon': 13.4050},
                'timezone': 'Europe/Berlin',
                'base_temp': 7, 'temp_range': 14,
                'base_aqi': 35, 'aqi_range': 25,
                'base_humidity': 73, 'humidity_range': 22,
                'pollution_level': 0.35, 'green_space': 0.44
            },
            'stockholm': {
                'name': 'Stockholm',
                'coordinates': {'lat': 59.3293, 'lon': 18.0686},
                'timezone': 'Europe/Stockholm',
                'base_temp': 4, 'temp_range': 12,
                'base_aqi': 18, 'aqi_range': 15,
                'base_humidity': 74, 'humidity_range': 20,
                'pollution_level': 0.15, 'green_space': 0.58
            },
            # Indian Cities
            'mumbai': {
                'name': 'Mumbai',
                'coordinates': {'lat': 19.0760, 'lon': 72.8777},
                'timezone': 'Asia/Kolkata',
                'base_temp': 27, 'temp_range': 8,
                'base_aqi': 145, 'aqi_range': 50,
                'base_humidity': 75, 'humidity_range': 20,
                'pollution_level': 0.78, 'green_space': 0.16
            },
            'delhi': {
                'name': 'Delhi',
                'coordinates': {'lat': 28.7041, 'lon': 77.1025},
                'timezone': 'Asia/Kolkata',
                'base_temp': 25, 'temp_range': 20,
                'base_aqi': 165, 'aqi_range': 60,
                'base_humidity': 65, 'humidity_range': 30,
                'pollution_level': 0.85, 'green_space': 0.22
            },
            'bangalore': {
                'name': 'Bangalore',
                'coordinates': {'lat': 12.9716, 'lon': 77.5946},
                'timezone': 'Asia/Kolkata',
                'base_temp': 24, 'temp_range': 8,
                'base_aqi': 95, 'aqi_range': 40,
                'base_humidity': 65, 'humidity_range': 25,
                'pollution_level': 0.58, 'green_space': 0.31
            },
            # Middle East & Asia
            'dubai': {
                'name': 'Dubai',
                'coordinates': {'lat': 25.2048, 'lon': 55.2708},
                'timezone': 'Asia/Dubai',
                'base_temp': 32, 'temp_range': 12,
                'base_aqi': 85, 'aqi_range': 35,
                'base_humidity': 55, 'humidity_range': 30,
                'pollution_level': 0.48, 'green_space': 0.12
            },
            'seoul': {
                'name': 'Seoul',
                'coordinates': {'lat': 37.5665, 'lon': 126.9780},
                'timezone': 'Asia/Seoul',
                'base_temp': 12, 'temp_range': 25,
                'base_aqi': 55, 'aqi_range': 30,
                'base_humidity': 68, 'humidity_range': 25,
                'pollution_level': 0.42, 'green_space': 0.28
            },
            'hong-kong': {
                'name': 'Hong Kong',
                'coordinates': {'lat': 22.3193, 'lon': 114.1694},
                'timezone': 'Asia/Hong_Kong',
                'base_temp': 23, 'temp_range': 12,
                'base_aqi': 65, 'aqi_range': 25,
                'base_humidity': 78, 'humidity_range': 20,
                'pollution_level': 0.45, 'green_space': 0.24
            },
            # Europe
            'zurich': {
                'name': 'Zurich',
                'coordinates': {'lat': 47.3769, 'lon': 8.5417},
                'timezone': 'Europe/Zurich',
                'base_temp': 9, 'temp_range': 18,
                'base_aqi': 12, 'aqi_range': 10,
                'base_humidity': 71, 'humidity_range': 20,
                'pollution_level': 0.12, 'green_space': 0.62
            },
            'oslo': {
                'name': 'Oslo',
                'coordinates': {'lat': 59.9139, 'lon': 10.7522},
                'timezone': 'Europe/Oslo',
                'base_temp': 5, 'temp_range': 15,
                'base_aqi': 8, 'aqi_range': 8,
                'base_humidity': 73, 'humidity_range': 20,
                'pollution_level': 0.08, 'green_space': 0.68
            },
            'reykjavik': {
                'name': 'Reykjavik',
                'coordinates': {'lat': 64.1466, 'lon': -21.9426},
                'timezone': 'Atlantic/Reykjavik',
                'base_temp': 4, 'temp_range': 10,
                'base_aqi': 5, 'aqi_range': 5,
                'base_humidity': 76, 'humidity_range': 15,
                'pollution_level': 0.05, 'green_space': 0.75
            },
            # Other Continents
            'cape-town': {
                'name': 'Cape Town',
                'coordinates': {'lat': -33.9249, 'lon': 18.4241},
                'timezone': 'Africa/Johannesburg',
                'base_temp': 18, 'temp_range': 10,
                'base_aqi': 28, 'aqi_range': 20,
                'base_humidity': 68, 'humidity_range': 25,
                'pollution_level': 0.22, 'green_space': 0.52
            },
            'sao-paulo': {
                'name': 'São Paulo',
                'coordinates': {'lat': -23.5505, 'lon': -46.6333},
                'timezone': 'America/Sao_Paulo',
                'base_temp': 20, 'temp_range': 8,
                'base_aqi': 95, 'aqi_range': 40,
                'base_humidity': 70, 'humidity_range': 25,
                'pollution_level': 0.68, 'green_space': 0.18
            },
            'mexico-city': {
                'name': 'Mexico City',
                'coordinates': {'lat': 19.4326, 'lon': -99.1332},
                'timezone': 'America/Mexico_City',
                'base_temp': 18, 'temp_range': 8,
                'base_aqi': 125, 'aqi_range': 45,
                'base_humidity': 55, 'humidity_range': 25,
                'pollution_level': 0.72, 'green_space': 0.15
            },
            'cairo': {
                'name': 'Cairo',
                'coordinates': {'lat': 30.0444, 'lon': 31.2357},
                'timezone': 'Africa/Cairo',
                'base_temp': 23, 'temp_range': 15,
                'base_aqi': 135, 'aqi_range': 50,
                'base_humidity': 45, 'humidity_range': 25,
                'pollution_level': 0.75, 'green_space': 0.08
            },
            'istanbul': {
                'name': 'Istanbul',
                'coordinates': {'lat': 41.0082, 'lon': 28.9784},
                'timezone': 'Europe/Istanbul',
                'base_temp': 14, 'temp_range': 18,
                'base_aqi': 85, 'aqi_range': 35,
                'base_humidity': 68, 'humidity_range': 25,
                'pollution_level': 0.55, 'green_space': 0.20
            },
            'moscow': {
                'name': 'Moscow',
                'coordinates': {'lat': 55.7558, 'lon': 37.6176},
                'timezone': 'Europe/Moscow',
                'base_temp': 5, 'temp_range': 25,
                'base_aqi': 48, 'aqi_range': 25,
                'base_humidity': 70, 'humidity_range': 25,
                'pollution_level': 0.38, 'green_space': 0.35
            }
        }

        # Simulate sensor network
        self.sensors = {
            'air_quality_001': {'type': 'air_quality', 'location': 'downtown', 'status': 'active'},
            'energy_meter_001': {'type': 'energy', 'location': 'residential', 'status': 'active'},
            'water_sensor_001': {'type': 'water', 'location': 'industrial', 'status': 'active'},
            'noise_monitor_001': {'type': 'noise', 'location': 'commercial', 'status': 'active'},
            'biodiversity_cam_001': {'type': 'biodiversity', 'location': 'park', 'status': 'active'}
        }

        # Initialize city vital signs
        self.city_data = {
            'overall_health': random.uniform(0.65, 0.85),
            'air_quality': random.uniform(0.60, 0.80),
            'energy_efficiency': random.uniform(0.70, 0.90),
            'water_efficiency': random.uniform(0.60, 0.85),
            'carbon_footprint': random.uniform(0.55, 0.75),
            'biodiversity_index': random.uniform(0.60, 0.80),
            'noise_pollution': random.uniform(0.65, 0.85),
            'last_updated': datetime.now()
        }

    def _get_time_factor(self, city='new-york'):
        """Generate time-based variation factor"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second

        # Daily cycle (rush hours have higher pollution)
        daily_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            daily_factor = 1.3
        elif 22 <= hour or hour <= 5:  # Night hours
            daily_factor = 0.7

        # Minute-based micro-variations (for real-time feel)
        micro_variation = 0.95 + (minute + second/60) * 0.001

        # Weekly cycle (weekends are cleaner)
        weekday = now.weekday()
        week_factor = 0.8 if weekday >= 5 else 1.0  # Weekend vs weekday

        return daily_factor * micro_variation * week_factor

    def _get_seasonal_factor(self):
        """Generate seasonal variation factor"""
        now = datetime.now()
        month = now.month
        day_of_year = now.timetuple().tm_yday

        # Seasonal pollution patterns (winter higher, summer lower)
        seasonal_base = 1.0 + 0.3 * math.cos((day_of_year - 180) * 2 * math.pi / 365)
        return max(0.7, min(1.4, seasonal_base))

    def get_city_stats(self, city='new-york'):
        """Get current city statistics for specific city with dynamic time-based variations"""
        config = self.city_configs.get(city, self.city_configs['new-york'])

        # Apply time-based and seasonal variations
        time_factor = self._get_time_factor(city)
        seasonal_factor = self._get_seasonal_factor()

        # Generate dynamic variations based on city characteristics and time
        pollution_multiplier = config['pollution_level'] * time_factor * seasonal_factor
        green_space_benefit = config['green_space'] * (1.1 - time_factor * 0.1)

        # Create unique hash based on current time for consistent but changing values
        time_seed = int(datetime.now().timestamp()) % 3600  # Changes every hour
        random.seed(time_seed + hash(city) % 1000)

        stats = {
            'city_name': config['name'],
            'coordinates': config['coordinates'],
            'overall_health': max(0.0, min(1.0, 0.85 - pollution_multiplier * 0.6 + random.uniform(-0.08, 0.08))),
            'air_quality': max(0.0, min(1.0, 0.92 - pollution_multiplier * 0.7 + random.uniform(-0.12, 0.12))),
            'energy_efficiency': max(0.0, min(1.0, 0.78 + green_space_benefit * 0.8 + random.uniform(-0.09, 0.09))),
            'water_efficiency': max(0.0, min(1.0, 0.73 + green_space_benefit * 0.6 + random.uniform(-0.11, 0.11))),
            'carbon_footprint': max(0.0, min(1.0, 0.82 - pollution_multiplier * 0.5 + random.uniform(-0.07, 0.07))),
            'biodiversity_index': max(0.0, min(1.0, green_space_benefit + 0.35 + random.uniform(-0.13, 0.13))),
            'noise_pollution': max(0.0, min(1.0, 0.88 - pollution_multiplier * 0.4 + random.uniform(-0.10, 0.10))),
            'last_updated': datetime.now(),
            'time_factor': round(time_factor, 3),
            'seasonal_factor': round(seasonal_factor, 3)
        }
        return stats

    def get_sensor_data(self, city='new-york'):
        """Get current sensor readings for specific city with dynamic variations"""
        config = self.city_configs.get(city, self.city_configs['new-york'])
        sensor_data = []

        # Get dynamic factors
        time_factor = self._get_time_factor(city)
        seasonal_factor = self._get_seasonal_factor()

        # Time-based seed for consistent but changing values
        time_seed = int(datetime.now().timestamp()) % 1800  # Changes every 30 minutes

        sensor_types = {
            'air_quality': {
                'base': 0.85 - config['pollution_level'] * 0.7,
                'time_sensitivity': 0.3,
                'seasonal_sensitivity': 0.2
            },
            'energy': {
                'base': 0.75 + config['green_space'] * 0.4,
                'time_sensitivity': 0.4,  # Energy varies more with time
                'seasonal_sensitivity': 0.1
            },
            'water': {
                'base': 0.70 + config['green_space'] * 0.3,
                'time_sensitivity': 0.1,
                'seasonal_sensitivity': 0.3
            },
            'noise': {
                'base': 0.80 - config['pollution_level'] * 0.5,
                'time_sensitivity': 0.5,  # Noise varies significantly with time
                'seasonal_sensitivity': 0.05
            },
            'biodiversity': {
                'base': config['green_space'] + 0.25,
                'time_sensitivity': 0.15,
                'seasonal_sensitivity': 0.4  # Biodiversity varies with seasons
            }
        }

        for sensor_id, sensor_info in self.sensors.items():
            sensor_type = sensor_info['type']
            type_config = sensor_types.get(sensor_type, sensor_types['air_quality'])

            # Calculate dynamic base value
            base_value = type_config['base']
            time_variation = (time_factor - 1.0) * type_config['time_sensitivity']
            seasonal_variation = (seasonal_factor - 1.0) * type_config['seasonal_sensitivity']

            # Unique seed per sensor for consistent variation
            random.seed(time_seed + hash(f"{sensor_id}_{city}") % 1000)

            final_value = base_value + time_variation + seasonal_variation + random.uniform(-0.15, 0.15)
            quality_value = max(0.65, min(0.99, final_value + random.uniform(-0.08, 0.08)))

            reading = {
                'id': f"{sensor_id}_{city}",
                'type': sensor_info['type'],
                'location': sensor_info['location'],
                'city': config['name'],
                'coordinates': config['coordinates'],
                'status': sensor_info['status'],
                'timestamp': datetime.now().isoformat(),
                'value': round(max(0.25, min(0.99, final_value)), 3),
                'quality_index': round(quality_value, 2),
                'time_factor': round(time_factor, 3),
                'seasonal_factor': round(seasonal_factor, 3)
            }
            sensor_data.append(reading)
        return sensor_data

    def get_predictions(self, city='new-york'):
        """Get AI predictions for specific city"""
        config = self.city_configs.get(city, self.city_configs['new-york'])
        predictions = []
        base_value = 0.8 - config['pollution_level']

        for hour in range(1, 7):
            # Simulate realistic prediction with decreasing confidence over time
            prediction_value = base_value + random.uniform(-0.1, 0.1)
            confidence = max(0.6, 0.95 - (hour * 0.05))

            predictions.append({
                'hour': hour,
                'predicted_health': round(max(0.0, min(1.0, prediction_value)), 3),
                'confidence': round(confidence, 3)
            })

        return predictions

    def get_recommendations(self):
        """Get regenerative recommendations"""
        recommendations = [
            "Implement rainwater harvesting system",
            "Install solar panels on government buildings",
            "Create green corridors connecting parks",
            "Upgrade to LED street lighting",
            "Establish urban farming initiatives",
            "Improve public transportation efficiency",
            "Plant native species in empty lots",
            "Install smart irrigation systems"
        ]

        # Return 3-5 random recommendations
        num_recs = random.randint(3, 5)
        return random.sample(recommendations, num_recs)

class RegeneraXHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for RegeneraX API endpoints"""

    def __init__(self, *args, server_instance=None, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        city = query_params.get('city', ['new-york'])[0]

        # API endpoints
        if path == '/api/city-stats':
            self.send_json_response(self.server_instance.get_city_stats(city))
        elif path == '/api/sensors':
            self.send_json_response(self.server_instance.get_sensor_data(city))
        elif path == '/api/predictions':
            self.send_json_response(self.server_instance.get_predictions(city))
        elif path == '/api/recommendations':
            self.send_json_response(self.server_instance.get_recommendations(city))
        elif path == '/api/environmental-metrics':
            metrics = asyncio.run(self.server_instance.environmental_system.get_current_metrics(city))
            self.send_json_response(metrics.__dict__ if metrics else {})
        elif path == '/api/weather-data':
            weather = asyncio.run(self.server_instance.environmental_system.get_weather_data(city))
            self.send_json_response(weather.__dict__ if weather else {})
        elif path == '/api/climate-recommendations':
            recommendations = asyncio.run(self.server_instance.environmental_system.get_recommendations(city))
            self.send_json_response([rec.__dict__ for rec in recommendations])
        elif path == '/api/iot-sensors':
            sensors = asyncio.run(self.server_instance.environmental_system.get_iot_sensors(city))
            self.send_json_response(sensors)
        elif path == '/api/iot-sensors':
            sensor_data = asyncio.run(self.server_instance.environmental_system.get_sensor_data())
            response_data = {}
            for sensor_id, readings in sensor_data.items():
                response_data[sensor_id] = [reading.__dict__ for reading in readings[-10:]]  # Last 10 readings
            self.send_json_response(response_data)
        elif path.startswith('/api/historical-data'):
            query_params = parse_qs(parsed_path.query)
            hours = int(query_params.get('hours', [24])[0])
            historical = asyncio.run(self.server_instance.environmental_system.get_historical_data(hours))
            # Convert dataclasses to dicts
            response_data = {
                'sensor_data': {},
                'weather_data': [],
                'time_range': historical['time_range']
            }
            for sensor_id, readings in historical['sensor_data'].items():
                response_data['sensor_data'][sensor_id] = [r.__dict__ for r in readings]
            for weather in historical['weather_data']:
                response_data['weather_data'].append(weather.__dict__)
            self.send_json_response(response_data)
        elif path == '/api/status':
            self.send_json_response({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'uptime': str(datetime.now() - self.server_instance.start_time) if self.server_instance.start_time else '0',
                'environmental_system': 'active' if self.server_instance.environmental_system.running else 'inactive'
            })
        else:
            # Serve static files (HTML, CSS, JS)
            super().do_GET()

    def send_json_response(self, data):
        """Send JSON response"""
        response = json.dumps(data, default=str, indent=2)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        self.wfile.write(response.encode('utf-8'))

    def log_message(self, format, *args):
        """Override to reduce log noise"""
        if args and len(args) > 0 and isinstance(args[0], str):
            if not any(x in args[0] for x in ['/api/', 'favicon.ico']):
                super().log_message(format, *args)

def create_handler(server_instance):
    """Create a handler with the server instance"""
    def handler(*args, **kwargs):
        RegeneraXHTTPHandler(*args, server_instance=server_instance, **kwargs)
    return handler

async def initialize_environmental_system(regenerax):
    """Initialize the environmental monitoring system"""
    try:
        # Initialize without starting the background loop to avoid hanging
        logger.info("🌍 Initializing Real-Time Environmental System...")
        regenerax.environmental_system.running = True
        # Pre-generate some initial data
        await regenerax.environmental_system._initialize_iot_sensors()
        await regenerax.environmental_system._initialize_weather_apis()
        print("✅ Environmental monitoring system initialized")
    except Exception as e:
        print(f"⚠️ Warning: Environmental system initialization failed: {e}")
        # Create a fallback basic system
        regenerax.environmental_system.running = True

def main():
    """Main server function"""
    print("🌱 Starting RegeneraX Web Server...")

    # Create server instance (use alternate port if 8000 is unavailable)
    # Default port is 8000; switch to 8001 to avoid potential address conflicts
    regenerax = RegeneraXServer(port=9010)
    regenerax.start_time = datetime.now()

    # Initialize environmental system
    try:
        asyncio.run(initialize_environmental_system(regenerax))
    except Exception as e:
        print(f"⚠️ Environmental system initialization failed: {e}")

    # Change to visualization directory to serve static files
    import os
    os.chdir('/home/hafizas-pc/citysense/visualization')

    # Create HTTP server
    handler = create_handler(regenerax)

    class RegeneraXHTTPServer(socketserver.TCPServer):
        def __init__(self, server_address, RequestHandlerClass, server_instance):
            self.server_instance = server_instance
            self.allow_reuse_address = True
            super().__init__(server_address, RequestHandlerClass)

        def finish_request(self, request, client_address):
            self.RequestHandlerClass(request, client_address, self, server_instance=self.server_instance)

    with RegeneraXHTTPServer(("", regenerax.port), RegeneraXHTTPHandler, regenerax) as httpd:
        print(f"✅ RegeneraX Server running at http://localhost:{regenerax.port}")
        print(f"📊 Perfect Dashboard: http://localhost:{regenerax.port}/environmental-dashboard-perfect.html")
        print(f"📈 Environmental Dashboard: http://localhost:{regenerax.port}/environmental-dashboard.html")
        print(f"📋 Basic Dashboard: http://localhost:{regenerax.port}/dashboard.html")
        print(f"🔌 API Status: http://localhost:{regenerax.port}/api/status")
        print("\n🚀 Available API Endpoints:")
        print("   /api/city-stats - Current city vital signs")
        print("   /api/sensors - Live sensor data")
        print("   /api/predictions - AI predictions")
        print("   /api/recommendations - Regenerative recommendations")
        print("   /api/environmental-metrics - Real-time environmental data")
        print("   /api/weather-data - Current weather information")
        print("   /api/climate-recommendations - Climate-responsive design recommendations")
        print("   /api/iot-sensors - IoT sensor readings")
        print("   /api/historical-data?hours=24 - Historical environmental data")
        print("   /api/status - Server status")
        print("\n⚡ Press Ctrl+C to stop the server")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down RegeneraX Server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()