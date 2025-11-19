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

    def __init__(self, port=9001):
        self.port = port
        self.sensors = {}
        self.city_data = {}
        self.running = False
        self.start_time = None
        self.environmental_system = RealTimeEnvironmentalSystem()
        self.initialize_data()

    def initialize_data(self):
        """Initialize demo data"""
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

    def get_city_stats(self):
        """Get current city statistics"""
        # Add some realistic variation
        base_time = time.time()
        for key in self.city_data:
            if key != 'last_updated':
                # Gentle oscillation based on time
                variation = 0.05 * (0.5 + 0.5 * random.random())
                self.city_data[key] = max(0.0, min(1.0,
                    self.city_data[key] + variation * (random.random() - 0.5)))

        self.city_data['last_updated'] = datetime.now()
        return self.city_data

    def get_sensor_data(self):
        """Get current sensor readings"""
        sensor_data = []
        for sensor_id, sensor_info in self.sensors.items():
            reading = {
                'id': sensor_id,
                'type': sensor_info['type'],
                'location': sensor_info['location'],
                'status': sensor_info['status'],
                'timestamp': datetime.now().isoformat(),
                'value': round(random.uniform(0.5, 0.95), 3),
                'quality_index': round(random.uniform(0.7, 0.98), 2)
            }
            sensor_data.append(reading)
        return sensor_data

    def get_predictions(self):
        """Get AI predictions"""
        predictions = []
        base_value = self.city_data['overall_health']

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

        # API endpoints
        if path == '/api/city-stats':
            self.send_json_response(self.server_instance.get_city_stats())
        elif path == '/api/sensors':
            self.send_json_response(self.server_instance.get_sensor_data())
        elif path == '/api/predictions':
            self.send_json_response(self.server_instance.get_predictions())
        elif path == '/api/recommendations':
            self.send_json_response(self.server_instance.get_recommendations())
        elif path == '/api/environmental-metrics':
            metrics = asyncio.run(self.server_instance.environmental_system.get_current_metrics())
            self.send_json_response(metrics.__dict__ if metrics else {})
        elif path == '/api/weather-data':
            weather = asyncio.run(self.server_instance.environmental_system.get_weather_data())
            self.send_json_response(weather.__dict__ if weather else {})
        elif path == '/api/climate-recommendations':
            recommendations = asyncio.run(self.server_instance.environmental_system.get_recommendations())
            self.send_json_response([rec.__dict__ for rec in recommendations])
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