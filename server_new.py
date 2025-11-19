#!/usr/bin/env python3
"""
RegeneraX Server - Web server with API endpoints for the dashboard
Provides both the demo functionality and web API endpoints
"""

import asyncio
import json
import logging
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

    def __init__(self, port=8001):
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

    def get_city_stats(self, city='new-york'):
        """Get current city statistics for specific city"""
        config = self.city_configs.get(city, self.city_configs['new-york'])

        # Generate city-specific stats based on configuration
        stats = {
            'city_name': config['name'],
            'coordinates': config['coordinates'],
            'overall_health': max(0.0, min(1.0, 0.8 - config['pollution_level'] + random.uniform(-0.1, 0.1))),
            'air_quality': max(0.0, min(1.0, 0.9 - config['pollution_level'] + random.uniform(-0.1, 0.1))),
            'energy_efficiency': max(0.0, min(1.0, 0.75 + config['green_space'] + random.uniform(-0.1, 0.1))),
            'water_efficiency': max(0.0, min(1.0, 0.7 + config['green_space'] * 0.5 + random.uniform(-0.1, 0.1))),
            'carbon_footprint': max(0.0, min(1.0, 0.8 - config['pollution_level'] + random.uniform(-0.05, 0.05))),
            'biodiversity_index': max(0.0, min(1.0, config['green_space'] + 0.3 + random.uniform(-0.1, 0.1))),
            'noise_pollution': max(0.0, min(1.0, 0.85 - config['pollution_level'] + random.uniform(-0.1, 0.1))),
            'last_updated': datetime.now()
        }
        return stats

    def get_sensor_data(self, city='new-york'):
        """Get current sensor readings for specific city"""
        config = self.city_configs.get(city, self.city_configs['new-york'])
        sensor_data = []

        for sensor_id, sensor_info in self.sensors.items():
            # Generate city-specific sensor values
            base_value = 0.8 - config['pollution_level'] * 0.5
            reading = {
                'id': f"{sensor_id}_{city}",
                'type': sensor_info['type'],
                'location': sensor_info['location'],
                'city': config['name'],
                'coordinates': config['coordinates'],
                'status': sensor_info['status'],
                'timestamp': datetime.now().isoformat(),
                'value': round(max(0.3, min(0.98, base_value + random.uniform(-0.2, 0.2))), 3),
                'quality_index': round(max(0.6, min(0.98, base_value + random.uniform(-0.1, 0.1))), 2)
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
        await regenerax.environmental_system.initialize()
        print("✅ Environmental monitoring system initialized")
    except Exception as e:
        print(f"⚠️ Warning: Environmental system initialization failed: {e}")

def main():
    """Main server function"""
    print("🌱 Starting RegeneraX Web Server...")

    # Create server instance
    regenerax = RegeneraXServer()
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