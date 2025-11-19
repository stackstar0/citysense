"""
RegeneraX VR/AR Interface System
===============================

Immersive virtual and augmented reality interfaces for city planning and design.
This system provides spatial interaction with urban data, 3D modeling capabilities,
and immersive visualization of regenerative design impacts.
"""

import asyncio
import json
import logging
import statistics
import math
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import websockets
import sqlite3

# Import our core systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.city_brain import CityBrain
from core.data_processor import DataProcessor
from visualization.websocket_manager import WebSocketManager

@dataclass
class VRSession:
    """VR session information"""
    session_id: str
    user_id: str
    headset_type: str  # 'oculus', 'vive', 'hololens', 'webxr'
    tracking_data: Dict[str, Any]
    interaction_mode: str  # 'design', 'analysis', 'simulation'
    current_scene: str
    tools_enabled: List[str]

@dataclass
class SpatialObject:
    """3D spatial object in VR environment"""
    object_id: str
    object_type: str  # 'building', 'tree', 'sensor', 'data_viz', 'tool'
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float]  # euler angles
    scale: Tuple[float, float, float]
    properties: Dict[str, Any]
    interactive: bool = True

class RegenerativeVREngine:
    """
    Core VR engine for regenerative city design and analysis
    Provides spatial interaction with urban data and design tools
    """

    def __init__(self):
        self.city_brain = None
        self.data_processor = None
        self.websocket_manager = None

        # VR Environment State
        self.active_sessions: Dict[str, VRSession] = {}
        self.spatial_objects: Dict[str, SpatialObject] = {}
        self.environmental_layers: Dict[str, Dict[str, Any]] = {}

        # Design Tools
        self.design_tools = {
            'building_creator': {
                'name': 'Regenerative Building Designer',
                'description': 'Create and modify buildings with real-time impact analysis',
                'capabilities': ['design', 'material_selection', 'energy_modeling', 'impact_preview']
            },
            'ecosystem_painter': {
                'name': 'Ecosystem Integration Tool',
                'description': 'Paint vegetation, water features, and natural systems',
                'capabilities': ['vegetation', 'water_features', 'soil_modification', 'habitat_creation']
            },
            'flow_visualizer': {
                'name': 'Resource Flow Visualizer',
                'description': 'Visualize energy, water, and material flows in 3D space',
                'capabilities': ['energy_flows', 'water_flows', 'material_flows', 'carbon_flows']
            },
            'time_simulator': {
                'name': 'Temporal Design Simulator',
                'description': 'See design impacts across different time scales',
                'capabilities': ['seasonal_changes', 'growth_simulation', 'aging_effects', 'climate_adaptation']
            },
            'impact_analyzer': {
                'name': 'Real-time Impact Analyzer',
                'description': 'Analyze regenerative potential of design decisions',
                'capabilities': ['carbon_analysis', 'biodiversity_impact', 'water_impact', 'energy_impact']
            }
        }

        # Environmental data layers that can be visualized in VR
        self.data_layers = {
            'air_quality': {
                'visualization': 'particle_field',
                'color_mapping': 'pollution_heatmap',
                'interaction': 'hover_for_details'
            },
            'energy_flows': {
                'visualization': 'flowing_particles',
                'color_mapping': 'energy_type',
                'interaction': 'trace_flow_paths'
            },
            'water_systems': {
                'visualization': 'fluid_simulation',
                'color_mapping': 'water_quality',
                'interaction': 'flow_modification'
            },
            'biodiversity': {
                'visualization': 'habitat_zones',
                'color_mapping': 'species_richness',
                'interaction': 'species_information'
            },
            'carbon_flows': {
                'visualization': 'gradient_fields',
                'color_mapping': 'sequestration_sources',
                'interaction': 'carbon_pathways'
            },
            'microclimate': {
                'visualization': 'weather_particles',
                'color_mapping': 'temperature_humidity',
                'interaction': 'climate_modification'
            }
        }

        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the VR engine and core systems"""
        try:
            self.city_brain = CityBrain()
            self.data_processor = DataProcessor()
            self.websocket_manager = WebSocketManager()

            await self.city_brain.initialize()
            await self.data_processor.initialize()

            # Initialize default environmental layers
            await self.initialize_environmental_layers()

            self.logger.info("RegenerativeVREngine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize VR engine: {e}")
            raise

    async def initialize_environmental_layers(self):
        """Initialize environmental data layers for VR visualization"""
        try:
            # Get current city data
            vital_signs = await self.city_brain.get_vital_signs()
            recent_data = await self.data_processor.get_recent_data(hours=24)

            # Process data into spatial layers
            for layer_name, layer_config in self.data_layers.items():
                layer_data = self.process_layer_data(layer_name, recent_data, vital_signs)
                self.environmental_layers[layer_name] = {
                    'config': layer_config,
                    'data': layer_data,
                    'last_updated': datetime.now().isoformat(),
                    'active': True
                }

            self.logger.info("Environmental layers initialized")

        except Exception as e:
            self.logger.error(f"Error initializing environmental layers: {e}")

    def process_layer_data(self, layer_name: str, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Process raw data into spatial visualization format"""
        try:
            if layer_name == 'air_quality':
                # Create 3D air quality field
                return self.create_air_quality_field(recent_data, vital_signs)

            elif layer_name == 'energy_flows':
                # Create energy flow visualization
                return self.create_energy_flow_field(recent_data, vital_signs)

            elif layer_name == 'water_systems':
                # Create water system visualization
                return self.create_water_system_field(recent_data, vital_signs)

            elif layer_name == 'biodiversity':
                # Create biodiversity zones
                return self.create_biodiversity_field(vital_signs)

            elif layer_name == 'carbon_flows':
                # Create carbon flow visualization
                return self.create_carbon_flow_field(recent_data, vital_signs)

            elif layer_name == 'microclimate':
                # Create microclimate visualization
                return self.create_microclimate_field(recent_data, vital_signs)

            else:
                return {'error': f'Unknown layer: {layer_name}'}

        except Exception as e:
            self.logger.error(f"Error processing layer {layer_name}: {e}")
            return {'error': str(e)}

    def create_air_quality_field(self, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Create 3D air quality particle field"""
        # Simulate air quality data across city grid
        grid_size = 50
        # Generate air quality data
        base_quality = vital_signs.get('air_quality', 0.7)

        particles = []
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(10):
                    quality = base_quality + random.gauss(0, 0.2)
                    particles.append({
                        'position': [x * 10, y * 10, z * 5],
                        'quality': float(quality),
                        'color': self.quality_to_color(quality),
                        'size': max(0.1, 1.0 - quality)  # Worse quality = larger particles
                    })

        return {
            'type': 'particle_field',
            'particles': particles,
            'bounds': [0, 0, 0, grid_size * 10, grid_size * 10, 50],
            'legend': {
                'good': [0.0, 1.0, 0.0, 0.6],
                'moderate': [1.0, 1.0, 0.0, 0.6],
                'poor': [1.0, 0.0, 0.0, 0.6]
            }
        }

    def create_energy_flow_field(self, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Create energy flow visualization"""
        flows = []

        # Simulate energy sources and sinks
        sources = [
            {'position': [100, 100, 20], 'type': 'solar', 'output': 1000},
            {'position': [300, 200, 15], 'type': 'wind', 'output': 800},
            {'position': [150, 350, 5], 'type': 'geothermal', 'output': 600}
        ]

        sinks = [
            {'position': [200, 150, 10], 'type': 'residential', 'demand': 500},
            {'position': [250, 300, 25], 'type': 'commercial', 'demand': 800},
            {'position': [180, 280, 15], 'type': 'industrial', 'demand': 700}
        ]

        # Create flow paths between sources and sinks
        for source in sources:
            for sink in sinks:
                flow_strength = min(source['output'], sink['demand']) / 1000.0
                flows.append({
                    'start': source['position'],
                    'end': sink['position'],
                    'strength': flow_strength,
                    'type': source['type'],
                    'color': self.energy_type_to_color(source['type']),
                    'animated': True,
                    'particle_count': int(flow_strength * 100)
                })

        return {
            'type': 'flow_field',
            'flows': flows,
            'sources': sources,
            'sinks': sinks,
            'total_generation': sum(s['output'] for s in sources),
            'total_demand': sum(s['demand'] for s in sinks),
            'efficiency': vital_signs.get('energy_efficiency', 0.8)
        }

    def create_water_system_field(self, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Create water system visualization"""
        return {
            'type': 'water_system',
            'watersheds': [
                {
                    'boundary': [[0, 0], [100, 50], [150, 100], [120, 200], [50, 180], [0, 100]],
                    'flow_direction': [0.5, -0.3],
                    'quality': vital_signs.get('water_quality', 0.8),
                    'flow_rate': 150.0
                }
            ],
            'water_features': [
                {'type': 'river', 'path': [[0, 100], [50, 120], [100, 140], [200, 160]], 'width': 5},
                {'type': 'pond', 'center': [80, 180], 'radius': 15},
                {'type': 'bioswale', 'path': [[120, 100], [140, 130], [160, 140]], 'width': 3}
            ],
            'stormwater_infrastructure': [
                {'type': 'green_roof', 'position': [100, 100], 'area': 200},
                {'type': 'rain_garden', 'position': [150, 180], 'area': 50},
                {'type': 'permeable_pavement', 'area': 1000}
            ]
        }

    def create_biodiversity_field(self, vital_signs: Dict) -> Dict[str, Any]:
        """Create biodiversity visualization"""
        biodiversity_index = vital_signs.get('biodiversity_index', 0.6)

        # Create habitat zones
        habitats = []
        for i in range(20):
            habitat_quality = max(0, min(1, biodiversity_index + random.gauss(0, 0.15)))
            habitats.append({
                'position': [
                    random.uniform(0, 500),
                    random.uniform(0, 500),
                    random.uniform(0, 10)
                ],
                'radius': random.uniform(10, 50),
                'quality': float(habitat_quality),
                'species_count': int(habitat_quality * 50),
                'habitat_type': random.choice(['forest', 'grassland', 'wetland', 'urban_park']),
                'connectivity': random.uniform(0.3, 1.0)
            })

        return {
            'type': 'biodiversity_field',
            'habitats': habitats,
            'corridors': self.generate_wildlife_corridors(habitats),
            'overall_index': biodiversity_index,
            'species_richness': int(biodiversity_index * 200)
        }

    def create_carbon_flow_field(self, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Create carbon flow visualization"""
        carbon_efficiency = vital_signs.get('carbon_footprint', 0.7)  # Lower is better

        # Carbon sources (emissions)
        sources = [
            {'position': [100, 200, 0], 'type': 'transportation', 'emission_rate': 50},
            {'position': [300, 150, 0], 'type': 'buildings', 'emission_rate': 80},
            {'position': [200, 300, 0], 'type': 'industry', 'emission_rate': 120}
        ]

        # Carbon sinks (sequestration)
        sinks = [
            {'position': [50, 100, 0], 'type': 'forest', 'sequestration_rate': 30},
            {'position': [250, 250, 0], 'type': 'green_roof', 'sequestration_rate': 15},
            {'position': [180, 180, 0], 'type': 'soil', 'sequestration_rate': 25}
        ]

        return {
            'type': 'carbon_field',
            'sources': sources,
            'sinks': sinks,
            'net_emission': sum(s['emission_rate'] for s in sources) - sum(s['sequestration_rate'] for s in sinks),
            'sequestration_potential': sum(s['sequestration_rate'] for s in sinks),
            'visualization_particles': self.generate_carbon_particles(sources, sinks)
        }

    def create_microclimate_field(self, recent_data: List[Dict], vital_signs: Dict) -> Dict[str, Any]:
        """Create microclimate visualization"""
        # Simulate temperature and humidity variations
        grid_size = 30
        # Generate temperature and humidity fields using simple random
        temperature_field = []
        humidity_field = []
        for x in range(grid_size):
            temp_row = []
            humid_row = []
            for y in range(grid_size):
                temp_row.append(22 + random.gauss(0, 3))
                humid_row.append(60 + random.gauss(0, 10))
            temperature_field.append(temp_row)
            humidity_field.append(humid_row)

        microclimates = []
        for x in range(grid_size):
            for y in range(grid_size):
                temp_val = temperature_field[x][y]
                humid_val = humidity_field[x][y]
                microclimates.append({
                    'position': [x * 15, y * 15, 5],
                    'temperature': float(temp_val),
                    'humidity': float(humid_val),
                    'comfort_index': self.calculate_comfort_index(temp_val, humid_val)
                })

        return {
            'type': 'microclimate_field',
            'microclimates': microclimates,
            'average_temperature': float(sum(sum(row) for row in temperature_field) / (len(temperature_field) * len(temperature_field[0]))),
            'average_humidity': float(sum(sum(row) for row in humidity_field) / (len(humidity_field) * len(humidity_field[0]))),
            'comfort_zones': self.identify_comfort_zones(microclimates)
        }

    # Utility methods for VR visualization
    def quality_to_color(self, quality: float) -> List[float]:
        """Convert quality value to RGBA color"""
        if quality > 0.8:
            return [0.0, 1.0, 0.0, 0.6]  # Green for good
        elif quality > 0.5:
            return [1.0, 1.0, 0.0, 0.6]  # Yellow for moderate
        else:
            return [1.0, 0.0, 0.0, 0.6]  # Red for poor

    def energy_type_to_color(self, energy_type: str) -> List[float]:
        """Convert energy type to color"""
        colors = {
            'solar': [1.0, 1.0, 0.0, 0.8],    # Yellow
            'wind': [0.0, 0.8, 1.0, 0.8],     # Light blue
            'geothermal': [1.0, 0.5, 0.0, 0.8], # Orange
            'grid': [0.5, 0.5, 0.5, 0.8]      # Gray
        }
        return colors.get(energy_type, [1.0, 1.0, 1.0, 0.8])

    def generate_wildlife_corridors(self, habitats: List[Dict]) -> List[Dict]:
        """Generate wildlife corridors between habitats"""
        corridors = []
        for i, habitat1 in enumerate(habitats):
            for j, habitat2 in enumerate(habitats[i+1:], i+1):
                distance = math.sqrt(sum((a - b)**2 for a, b in zip(habitat1['position'], habitat2['position'])))
                if distance < 100 and habitat1['connectivity'] > 0.5 and habitat2['connectivity'] > 0.5:
                    corridors.append({
                        'start': habitat1['position'],
                        'end': habitat2['position'],
                        'width': min(habitat1['quality'], habitat2['quality']) * 10,
                        'quality': (habitat1['quality'] + habitat2['quality']) / 2
                    })
        return corridors

    def generate_carbon_particles(self, sources: List[Dict], sinks: List[Dict]) -> List[Dict]:
        """Generate carbon flow particles"""
        particles = []

        # Emission particles (red, moving up)
        for source in sources:
            for _ in range(source['emission_rate'] // 10):
                particles.append({
                    'position': source['position'],
                    'velocity': [
                        random.gauss(0, 2),
                        random.gauss(0, 2),
                        random.uniform(1, 3)
                    ],
                    'color': [1.0, 0.2, 0.2, 0.6],
                    'lifetime': 10.0,
                    'type': 'emission'
                })

        # Sequestration particles (green, moving down)
        for sink in sinks:
            for _ in range(sink['sequestration_rate'] // 5):
                particles.append({
                    'position': [sink['position'][0], sink['position'][1], sink['position'][2] + 10],
                    'velocity': [
                        random.gauss(0, 1),
                        random.gauss(0, 1),
                        random.uniform(-2, -0.5)
                    ],
                    'color': [0.2, 1.0, 0.2, 0.6],
                    'lifetime': 8.0,
                    'type': 'sequestration'
                })

        return particles

    def calculate_comfort_index(self, temperature: float, humidity: float) -> float:
        """Calculate human comfort index from temperature and humidity"""
        # Simplified comfort calculation
        optimal_temp = 22.0
        optimal_humidity = 50.0

        temp_score = 1.0 - abs(temperature - optimal_temp) / 15.0
        humidity_score = 1.0 - abs(humidity - optimal_humidity) / 30.0

        return max(0.0, min(1.0, (temp_score + humidity_score) / 2.0))

    def identify_comfort_zones(self, microclimates: List[Dict]) -> List[Dict]:
        """Identify areas of high comfort"""
        comfort_zones = []

        # Group nearby high-comfort areas
        high_comfort = [m for m in microclimates if m['comfort_index'] > 0.7]

        # Simple clustering of comfort zones
        for climate in high_comfort:
            comfort_zones.append({
                'center': climate['position'],
                'radius': 20,
                'comfort_level': climate['comfort_index'],
                'temperature': climate['temperature'],
                'humidity': climate['humidity']
            })

        return comfort_zones

    # VR Session Management
    async def create_vr_session(self, user_id: str, headset_type: str, interaction_mode: str) -> str:
        """Create a new VR session"""
        import uuid
        session_id = str(uuid.uuid4())

        session = VRSession(
            session_id=session_id,
            user_id=user_id,
            headset_type=headset_type,
            tracking_data={},
            interaction_mode=interaction_mode,
            current_scene='city_overview',
            tools_enabled=list(self.design_tools.keys())
        )

        self.active_sessions[session_id] = session

        # Send initial scene data
        await self.send_scene_data(session_id)

        self.logger.info(f"Created VR session {session_id} for user {user_id}")
        return session_id

    async def send_scene_data(self, session_id: str):
        """Send current scene data to VR client"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        scene_data = {
            'session_id': session_id,
            'scene': session.current_scene,
            'environmental_layers': {
                name: layer for name, layer in self.environmental_layers.items()
                if layer['active']
            },
            'spatial_objects': list(self.spatial_objects.values()),
            'available_tools': self.design_tools,
            'interaction_mode': session.interaction_mode
        }

        # Send via WebSocket (would need WebSocket connection in real implementation)
        await self.websocket_manager.broadcast(json.dumps({
            'type': 'scene_update',
            'data': scene_data
        }))

    async def handle_vr_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VR interaction event"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}

        session = self.active_sessions[session_id]
        interaction_type = interaction_data.get('type')

        try:
            if interaction_type == 'tool_select':
                return await self.handle_tool_selection(session, interaction_data)
            elif interaction_type == 'object_create':
                return await self.handle_object_creation(session, interaction_data)
            elif interaction_type == 'object_modify':
                return await self.handle_object_modification(session, interaction_data)
            elif interaction_type == 'layer_toggle':
                return await self.handle_layer_toggle(session, interaction_data)
            elif interaction_type == 'analysis_request':
                return await self.handle_analysis_request(session, interaction_data)
            else:
                return {'error': f'Unknown interaction type: {interaction_type}'}

        except Exception as e:
            self.logger.error(f"Error handling VR interaction: {e}")
            return {'error': str(e)}

    async def handle_tool_selection(self, session: VRSession, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool selection in VR"""
        tool_name = interaction_data.get('tool_name')

        if tool_name not in self.design_tools:
            return {'error': f'Unknown tool: {tool_name}'}

        tool_info = self.design_tools[tool_name]

        return {
            'success': True,
            'selected_tool': tool_name,
            'tool_info': tool_info,
            'instructions': f"You have selected the {tool_info['name']}. {tool_info['description']}"
        }

    async def handle_object_creation(self, session: VRSession, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle object creation in VR space"""
        object_type = interaction_data.get('object_type')
        position = interaction_data.get('position', [0, 0, 0])
        properties = interaction_data.get('properties', {})

        # Generate unique object ID
        import uuid
        object_id = str(uuid.uuid4())

        # Create spatial object
        spatial_object = SpatialObject(
            object_id=object_id,
            object_type=object_type,
            position=tuple(position),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
            properties=properties,
            interactive=True
        )

        self.spatial_objects[object_id] = spatial_object

        # If it's a building, analyze its regenerative potential
        analysis = {}
        if object_type == 'building':
            analysis = await self.analyze_building_impact(spatial_object)

        return {
            'success': True,
            'object_id': object_id,
            'object': spatial_object.__dict__,
            'analysis': analysis
        }

    async def analyze_building_impact(self, building: SpatialObject) -> Dict[str, Any]:
        """Analyze the regenerative impact of a building design"""
        properties = building.properties

        # Simulate regenerative impact analysis
        analysis = {
            'energy_performance': {
                'annual_consumption': properties.get('floor_area', 1000) * 50,  # kWh/year
                'renewable_generation': properties.get('solar_area', 0) * 200,  # kWh/year
                'net_energy': 0  # Will be calculated
            },
            'water_management': {
                'rainwater_capture': properties.get('roof_area', 500) * 0.8,  # liters/year
                'greywater_reuse': properties.get('occupants', 20) * 50 * 365,  # liters/year
                'stormwater_retention': properties.get('permeable_area', 200) * 100  # liters
            },
            'biodiversity_impact': {
                'habitat_created': properties.get('green_roof_area', 0) + properties.get('green_wall_area', 0),
                'species_supported': max(0, (properties.get('green_roof_area', 0) * 0.1)),
                'connectivity_score': 0.7  # Would be calculated based on nearby habitats
            },
            'carbon_impact': {
                'embodied_carbon': properties.get('floor_area', 1000) * 300,  # kg CO2
                'operational_carbon': properties.get('floor_area', 1000) * 30,  # kg CO2/year
                'sequestration_potential': properties.get('vegetation_carbon', 0)  # kg CO2/year
            }
        }

        # Calculate net energy
        analysis['energy_performance']['net_energy'] = (
            analysis['energy_performance']['renewable_generation'] -
            analysis['energy_performance']['annual_consumption']
        )

        # Calculate regenerative score
        regenerative_score = self.calculate_regenerative_score(analysis)
        analysis['regenerative_score'] = regenerative_score

        return analysis

    def calculate_regenerative_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall regenerative score for a design"""
        scores = []

        # Energy score (positive if net-positive)
        net_energy = analysis['energy_performance']['net_energy']
        energy_score = min(1.0, max(0.0, (net_energy + 5000) / 10000))
        scores.append(energy_score)

        # Water score
        water_captured = analysis['water_management']['rainwater_capture']
        water_score = min(1.0, water_captured / 5000)
        scores.append(water_score)

        # Biodiversity score
        habitat_score = min(1.0, analysis['biodiversity_impact']['habitat_created'] / 500)
        scores.append(habitat_score)

        # Carbon score (lower operational carbon is better)
        carbon_score = max(0.0, 1.0 - analysis['carbon_impact']['operational_carbon'] / 50000)
        scores.append(carbon_score)

        return sum(scores) / len(scores)

# WebXR Interface for browser-based VR
class WebXRInterface:
    """
    Browser-based WebXR interface for regenerative city design
    Provides VR capabilities without requiring dedicated headsets
    """

    def __init__(self):
        self.vr_engine = RegenerativeVREngine()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize WebXR interface"""
        await self.vr_engine.initialize()
        self.logger.info("WebXR interface initialized")

    def generate_webxr_scene(self) -> str:
        """Generate A-Frame WebXR scene HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>RegeneraX VR - Regenerative City Design</title>
    <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #ui-overlay {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            max-width: 300px;
        }
        .tool-button {
            display: inline-block;
            margin: 5px;
            padding: 8px 12px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        .tool-button:hover { background: #1976D2; }
        .layer-toggle {
            display: block;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div id="ui-overlay">
        <h3>🌱 RegeneraX VR Designer</h3>
        <div id="tools">
            <button class="tool-button" onclick="selectTool('building_creator')">🏢 Building Designer</button>
            <button class="tool-button" onclick="selectTool('ecosystem_painter')">🌿 Ecosystem Tool</button>
            <button class="tool-button" onclick="selectTool('flow_visualizer')">💧 Flow Visualizer</button>
            <button class="tool-button" onclick="selectTool('impact_analyzer')">📊 Impact Analyzer</button>
        </div>
        <div id="layers">
            <h4>Environmental Layers:</h4>
            <label class="layer-toggle"><input type="checkbox" checked onchange="toggleLayer('air_quality')"> Air Quality</label>
            <label class="layer-toggle"><input type="checkbox" checked onchange="toggleLayer('energy_flows')"> Energy Flows</label>
            <label class="layer-toggle"><input type="checkbox" onchange="toggleLayer('water_systems')"> Water Systems</label>
            <label class="layer-toggle"><input type="checkbox" onchange="toggleLayer('biodiversity')"> Biodiversity</label>
            <label class="layer-toggle"><input type="checkbox" onchange="toggleLayer('carbon_flows')"> Carbon Flows</label>
        </div>
    </div>

    <a-scene
        vr-mode-ui="enabled: true"
        embedded
        background="color: #87CEEB"
        fog="type: linear; color: #87CEEB; near: 100; far: 1000">

        <a-assets>
            <!-- 3D Models would be loaded here -->
            <a-mixin id="building" geometry="primitive: box" material="color: #FFF; opacity: 0.8"></a-mixin>
            <a-mixin id="tree" geometry="primitive: cylinder; height: 8; radiusTop: 0.5; radiusBottom: 1" material="color: #8B4513"></a-mixin>
        </a-assets>

        <!-- Lighting -->
        <a-light type="ambient" color="#404040" intensity="0.4"></a-light>
        <a-light type="directional" position="10 20 10" color="#FFF" intensity="0.6" castShadow="true"></a-light>

        <!-- Ground -->
        <a-plane
            position="0 0 0"
            rotation="-90 0 0"
            width="1000"
            height="1000"
            color="#7BC8A4"
            receiveShadow="true">
        </a-plane>

        <!-- City Grid -->
        <a-entity id="city-grid">
            <!-- Grid lines -->
            <a-entity id="grid-lines"></a-entity>

            <!-- Buildings -->
            <a-entity id="buildings">
                <a-box position="50 5 50" height="10" width="20" depth="30" color="#4CAF50" opacity="0.8"></a-box>
                <a-box position="100 8 80" height="16" width="25" depth="25" color="#2196F3" opacity="0.8"></a-box>
                <a-box position="150 6 120" height="12" width="30" depth="20" color="#FF9800" opacity="0.8"></a-box>
            </a-entity>

            <!-- Vegetation -->
            <a-entity id="vegetation">
                <a-cylinder position="30 4 30" height="8" radius="1" color="#228B22"></a-cylinder>
                <a-cylinder position="80 4 60" height="8" radius="1" color="#228B22"></a-cylinder>
                <a-cylinder position="130 4 90" height="8" radius="1" color="#228B22"></a-cylinder>
            </a-entity>
        </a-entity>

        <!-- Environmental Layers -->
        <a-entity id="air-quality-layer" visible="true">
            <!-- Air quality particles would be generated here -->
        </a-entity>

        <a-entity id="energy-flow-layer" visible="true">
            <!-- Energy flow visualization would be generated here -->
        </a-entity>

        <!-- VR Controllers -->
        <a-entity id="leftHand"
            hand-controls="hand: left; handModelStyle: lowPoly; color: #ffb400"
            teleport-controls="cameraRig: #rig; teleportOrigin: #camera; button: trigger; curveShootingKeys: trackpad; landingNormal: 0 1 0; landingMaxAngle: 45">
        </a-entity>

        <a-entity id="rightHand"
            hand-controls="hand: right; handModelStyle: lowPoly; color: #ffb400"
            raycaster="objects: .interactive"
            cursor="rayOrigin: mouse">
        </a-entity>

        <!-- Camera Rig -->
        <a-entity id="rig" movement-controls position="0 1.6 0">
            <a-camera id="camera"
                position="0 0 0"
                look-controls="pointerLockEnabled: true"
                wasd-controls="acceleration: 20"
                cursor="rayOrigin: mouse">
            </a-camera>
        </a-entity>

        <!-- UI Panels in VR -->
        <a-plane
            id="info-panel"
            position="2 2 -3"
            width="3"
            height="2"
            color="#000"
            opacity="0.7"
            text="value: RegeneraX City Designer\\nAnalyzing regenerative potential...\\n\\nNet Energy: +2,500 kWh/yr\\nCarbon Sequestered: 1,200 kg/yr\\nBiodiversity Index: 0.78\\n\\nRegeneration Score: 85%; align: center; color: #FFF">
        </a-plane>
    </a-scene>

    <script>
        let currentTool = null;
        let activeLayers = new Set(['air_quality', 'energy_flows']);

        function selectTool(toolName) {
            currentTool = toolName;
            console.log('Selected tool:', toolName);

            // Update info panel
            const infoPanel = document.querySelector('#info-panel');
            const toolDescriptions = {
                'building_creator': 'Building Designer\\nCreate regenerative structures\\nAnalyze energy & water systems',
                'ecosystem_painter': 'Ecosystem Integration\\nAdd vegetation & habitats\\nEnhance biodiversity',
                'flow_visualizer': 'Resource Flow Visualizer\\nView energy, water, carbon flows\\nOptimize resource efficiency',
                'impact_analyzer': 'Impact Analyzer\\nReal-time sustainability metrics\\nRegeneration potential scoring'
            };

            if (infoPanel) {
                infoPanel.setAttribute('text', 'value', toolDescriptions[toolName] || 'Tool selected');
            }
        }

        function toggleLayer(layerName) {
            if (activeLayers.has(layerName)) {
                activeLayers.delete(layerName);
            } else {
                activeLayers.add(layerName);
            }

            const layerElement = document.querySelector(`#${layerName.replace('_', '-')}-layer`);
            if (layerElement) {
                layerElement.setAttribute('visible', activeLayers.has(layerName));
            }

            console.log('Active layers:', Array.from(activeLayers));
        }

        // Initialize VR scene
        document.addEventListener('DOMContentLoaded', function() {
            const scene = document.querySelector('a-scene');

            scene.addEventListener('loaded', function() {
                console.log('RegeneraX VR scene loaded');
                selectTool('building_creator');
            });

            // Handle VR interactions
            scene.addEventListener('click', function(evt) {
                if (currentTool && evt.detail.intersection) {
                    const position = evt.detail.intersection.point;
                    console.log('VR interaction at:', position, 'with tool:', currentTool);

                    // This would trigger actual design actions
                    handleVRInteraction(currentTool, position);
                }
            });
        });

        function handleVRInteraction(tool, position) {
            // This would communicate with the Python backend
            const interaction = {
                type: 'tool_use',
                tool: tool,
                position: [position.x, position.y, position.z],
                timestamp: new Date().toISOString()
            };

            console.log('VR Interaction:', interaction);

            // Example: Create a building at click position
            if (tool === 'building_creator') {
                createBuildingAtPosition(position);
            }
        }

        function createBuildingAtPosition(position) {
            const buildings = document.querySelector('#buildings');
            const building = document.createElement('a-box');

            building.setAttribute('position', `${position.x} 5 ${position.z}`);
            building.setAttribute('height', '10');
            building.setAttribute('width', '15');
            building.setAttribute('depth', '15');
            building.setAttribute('color', '#4CAF50');
            building.setAttribute('opacity', '0.8');
            building.setAttribute('class', 'interactive');

            buildings.appendChild(building);

            console.log('Created building at:', position);
        }
    </script>
</body>
</html>
        """

# Example usage
async def main():
    """Example usage of the VR interface system"""
    print("=== RegeneraX VR Interface Demo ===\n")

    # Initialize VR engine
    vr_engine = RegenerativeVREngine()
    await vr_engine.initialize()

    # Create VR session
    session_id = await vr_engine.create_vr_session('user123', 'webxr', 'design')
    print(f"Created VR session: {session_id}")

    # Simulate VR interactions
    interactions = [
        {
            'type': 'tool_select',
            'tool_name': 'building_creator'
        },
        {
            'type': 'object_create',
            'object_type': 'building',
            'position': [100, 0, 100],
            'properties': {
                'floor_area': 2000,
                'solar_area': 500,
                'green_roof_area': 300,
                'occupants': 50
            }
        },
        {
            'type': 'analysis_request',
            'analysis_type': 'regenerative_potential'
        }
    ]

    for interaction in interactions:
        print(f"\n🎮 VR Interaction: {interaction['type']}")
        result = await vr_engine.handle_vr_interaction(session_id, interaction)
        print(f"Result: {json.dumps(result, indent=2)}")

    # Generate WebXR interface
    webxr = WebXRInterface()
    await webxr.initialize()

    print("\n🌐 WebXR Interface HTML generated")
    print("This would create an immersive VR experience accessible via web browsers")

if __name__ == "__main__":
    asyncio.run(main())