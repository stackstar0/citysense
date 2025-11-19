#!/usr/bin/env python3
"""
RegeneraX Demo - Simplified Version
A minimal working version of RegeneraX that demonstrates core concepts
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
import random
import statistics
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRegeneraX:
    """Simplified RegeneraX system that can run without heavy dependencies"""

    def __init__(self):
        self.sensors = {}
        self.city_data = {}
        self.running = False
        self.start_time = None

    async def initialize(self):
        """Initialize the simplified system"""
        logger.info("🌱 Initializing RegeneraX Demo System...")

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
            'overall_health': 0.78,
            'air_quality': 0.72,
            'energy_efficiency': 0.83,
            'water_efficiency': 0.71,
            'carbon_footprint': 0.65,
            'biodiversity_index': 0.69,
            'noise_pollution': 0.74,
            'last_updated': datetime.now()
        }

        self.running = True
        self.start_time = datetime.now()
        logger.info("✅ RegeneraX Demo initialized successfully!")

    async def get_vital_signs(self) -> Dict[str, Any]:
        """Get current city vital signs"""
        # Simulate slight variations in data
        for key in ['air_quality', 'energy_efficiency', 'water_efficiency', 'carbon_footprint', 'biodiversity_index']:
            if key in self.city_data:
                # Add small random variation (-0.02 to +0.02)
                variation = random.uniform(-0.02, 0.02)
                self.city_data[key] = max(0, min(1, self.city_data[key] + variation))

        # Update overall health as average
        health_metrics = [self.city_data[k] for k in ['air_quality', 'energy_efficiency', 'water_efficiency'] if k in self.city_data]
        self.city_data['overall_health'] = statistics.mean(health_metrics) if health_metrics else 0.5
        self.city_data['last_updated'] = datetime.now()

        return self.city_data.copy()

    async def get_sensor_data(self, sensor_id: str = None) -> List[Dict[str, Any]]:
        """Get sensor data"""
        data = []
        sensors_to_query = [sensor_id] if sensor_id else list(self.sensors.keys())

        for sid in sensors_to_query:
            if sid not in self.sensors:
                continue

            sensor = self.sensors[sid]
            # Generate simulated readings based on sensor type
            reading = {
                'sensor_id': sid,
                'sensor_type': sensor['type'],
                'location': sensor['location'],
                'timestamp': datetime.now(),
                'status': sensor['status']
            }

            # Add type-specific measurements
            if sensor['type'] == 'air_quality':
                reading['measurements'] = {
                    'pm25': random.uniform(5, 35),
                    'pm10': random.uniform(10, 50),
                    'co2': random.uniform(350, 450),
                    'quality_index': random.uniform(0.4, 0.9)
                }
            elif sensor['type'] == 'energy':
                reading['measurements'] = {
                    'consumption_kw': random.uniform(50, 200),
                    'generation_kw': random.uniform(20, 150),
                    'efficiency': random.uniform(0.6, 0.95)
                }
            elif sensor['type'] == 'water':
                reading['measurements'] = {
                    'flow_rate': random.uniform(10, 100),
                    'quality_index': random.uniform(0.7, 0.95),
                    'temperature': random.uniform(18, 25)
                }
            elif sensor['type'] == 'noise':
                reading['measurements'] = {
                    'decibel_level': random.uniform(35, 75),
                    'frequency_analysis': {'low': random.uniform(0.2, 0.6), 'high': random.uniform(0.1, 0.4)}
                }
            elif sensor['type'] == 'biodiversity':
                reading['measurements'] = {
                    'species_count': random.randint(5, 25),
                    'vegetation_index': random.uniform(0.3, 0.9),
                    'habitat_quality': random.uniform(0.4, 0.8)
                }

            data.append(reading)

        return data

    async def get_predictions(self, metric: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Generate simple predictions"""
        current_value = self.city_data.get(metric, 0.5)
        predictions = []

        for hour in range(1, hours_ahead + 1):
            # Simple trend simulation with some randomness
            trend = 0.001 * hour  # Slight improvement over time
            noise = random.uniform(-0.05, 0.05)
            predicted_value = max(0, min(1, current_value + trend + noise))

            predictions.append({
                'timestamp': datetime.now() + timedelta(hours=hour),
                'predicted_value': predicted_value,
                'confidence': random.uniform(0.7, 0.95)
            })

        return {
            'metric': metric,
            'current_value': current_value,
            'predictions': predictions,
            'model_type': 'trend_analysis',
            'generated_at': datetime.now()
        }

    async def analyze_regenerative_potential(self, design_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regenerative potential of a design proposal"""

        # Extract key parameters
        building_area = design_proposal.get('building_area', 1000)  # sq ft
        solar_panels = design_proposal.get('solar_panels', False)
        green_roof = design_proposal.get('green_roof', False)
        rainwater_harvesting = design_proposal.get('rainwater_harvesting', False)
        local_materials = design_proposal.get('local_materials', False)

        # Calculate regenerative scores
        energy_score = 0.3  # Base score
        if solar_panels:
            energy_score += 0.4
        if building_area < 2000:  # Smaller buildings are more efficient
            energy_score += 0.2

        water_score = 0.2  # Base score
        if rainwater_harvesting:
            water_score += 0.5
        if green_roof:
            water_score += 0.3

        biodiversity_score = 0.1  # Base score
        if green_roof:
            biodiversity_score += 0.6
        if building_area > 500:  # Larger buildings can support more biodiversity
            biodiversity_score += 0.2

        carbon_score = 0.2  # Base score
        if local_materials:
            carbon_score += 0.3
        if solar_panels:
            carbon_score += 0.4
        if green_roof:
            carbon_score += 0.1

        # Overall regenerative score
        overall_score = statistics.mean([energy_score, water_score, biodiversity_score, carbon_score])

        analysis = {
            'overall_regenerative_score': min(1.0, overall_score),
            'category_scores': {
                'energy': min(1.0, energy_score),
                'water': min(1.0, water_score),
                'biodiversity': min(1.0, biodiversity_score),
                'carbon': min(1.0, carbon_score)
            },
            'estimated_impacts': {
                'annual_energy_savings': building_area * 25 * (energy_score - 0.3),  # kWh
                'annual_water_savings': building_area * 10 * (water_score - 0.2),    # gallons
                'carbon_sequestration': building_area * 0.5 * biodiversity_score,    # kg CO2/year
                'biodiversity_improvement': biodiversity_score * 100                  # species index
            },
            'recommendations': []
        }

        # Generate recommendations
        if not solar_panels:
            analysis['recommendations'].append("Add solar panels to achieve net-positive energy")
        if not green_roof:
            analysis['recommendations'].append("Install green roof for water management and biodiversity")
        if not rainwater_harvesting:
            analysis['recommendations'].append("Implement rainwater harvesting system")
        if not local_materials:
            analysis['recommendations'].append("Use locally-sourced materials to reduce carbon footprint")

        if overall_score > 0.8:
            analysis['recommendations'].append("Excellent regenerative design! Consider sharing as a model.")

        return analysis

    async def chat_with_ai(self, user_input: str) -> str:
        """Simple conversational AI responses"""
        user_input_lower = user_input.lower()

        if any(word in user_input_lower for word in ['energy', 'solar', 'power']):
            return """**Energy Efficiency Guidance:**

Based on current city data, here are key recommendations for regenerative energy design:

• **Solar Integration**: Current city energy efficiency is 83%. Adding solar panels can push your building toward net-positive energy.

• **Passive Design**: Orient buildings to maximize winter sun and minimize summer heat gain.

• **Smart Systems**: Use building automation to optimize energy consumption patterns.

• **Community Energy**: Consider district-level energy sharing for maximum efficiency.

Would you like specific calculations for your project size?"""

        elif any(word in user_input_lower for word in ['water', 'rain', 'drainage']):
            return """**Water Management Guidance:**

Current city water efficiency: 71% - room for improvement!

Regenerative water strategies:
• **Rainwater Harvesting**: Capture and store precipitation for irrigation and non-potable uses
• **Greywater Systems**: Treat and reuse water from sinks and showers
• **Bioswales**: Natural drainage systems that filter stormwater while creating habitat
• **Permeable Surfaces**: Allow water infiltration to recharge groundwater

These systems can reduce municipal water demand by 40-60%."""

        elif any(word in user_input_lower for word in ['ecosystem', 'biodiversity', 'green', 'nature']):
            return """**Ecosystem Integration Guidance:**

Current biodiversity index: 69% - good foundation to build on!

Regenerative ecosystem strategies:
• **Native Plants**: Use local flora that supports existing wildlife food webs
• **Wildlife Corridors**: Connect your green spaces to nearby parks and natural areas
• **Pollinator Gardens**: Create habitat specifically for bees, butterflies, and other pollinators
• **Carbon Sequestration**: Maximize soil and plant-based carbon storage

Your design can actively contribute to urban ecosystem health!"""

        elif any(word in user_input_lower for word in ['cost', 'budget', 'money', 'roi']):
            return """**Economic Benefits of Regenerative Design:**

Regenerative features often pay for themselves:

• **Energy Systems**: Solar + efficiency typically pays back in 6-8 years
• **Water Systems**: Rainwater harvesting saves $500-2000/year for typical buildings
• **Green Infrastructure**: Increases property values by 10-15%
• **Operational Savings**: 20-40% reduction in utility costs

Plus: Avoid future costs from climate impacts, regulations, and resource scarcity."""

        elif any(word in user_input_lower for word in ['building', 'design', 'architecture']):
            return """**Regenerative Building Design Principles:**

Creating buildings that give back more than they take:

1. **Net-Positive Energy**: Generate more renewable energy than consumed
2. **Water Cycle Restoration**: Capture, clean, and infiltrate stormwater
3. **Ecosystem Integration**: Create habitat and support biodiversity
4. **Circular Materials**: Use recycled, renewable, and locally-sourced materials
5. **Adaptive Resilience**: Design for changing climate conditions

Start with the biggest impact strategies for your site and budget."""

        else:
            return """**Welcome to RegeneraX Regenerative Design Intelligence!**

I can help you create buildings and communities that improve rather than degrade natural systems.

Ask me about:
• **Energy systems** - achieving net-positive energy
• **Water management** - capturing and cleaning stormwater
• **Ecosystem integration** - supporting biodiversity
• **Economic benefits** - ROI of regenerative features
• **Design principles** - holistic regenerative strategies

What specific aspect of regenerative design interests you?"""

    async def run_demo(self):
        """Run an interactive demo"""
        if not self.running:
            await self.initialize()

        print("\n" + "="*60)
        print("🌱 RegeneraX - Regenerative Urban Intelligence Demo")
        print("="*60)

        # Show vital signs
        print("\n📊 Current City Vital Signs:")
        vital_signs = await self.get_vital_signs()
        for key, value in vital_signs.items():
            if key != 'last_updated' and isinstance(value, (int, float)):
                print(f"   {key.replace('_', ' ').title()}: {value:.1%}")

        # Show sensor data
        print("\n🔬 Sample Sensor Readings:")
        sensor_data = await self.get_sensor_data()
        for data in sensor_data[:3]:  # Show first 3 sensors
            print(f"   {data['sensor_id']} ({data['sensor_type']}): Active")
            if 'quality_index' in str(data['measurements']):
                for k, v in data['measurements'].items():
                    if 'quality' in k or 'index' in k:
                        print(f"     {k}: {v:.2f}")

        # Demonstrate AI predictions
        print("\n🔮 AI Predictions (Next 6 Hours):")
        predictions = await self.get_predictions('air_quality', 6)
        for i, pred in enumerate(predictions['predictions'][:3]):
            hour = i + 1
            confidence = pred['confidence']
            value = pred['predicted_value']
            print(f"   Hour {hour}: {value:.1%} (confidence: {confidence:.1%})")

        # Demonstrate regenerative analysis
        print("\n🏗️ Regenerative Design Analysis:")
        sample_building = {
            'building_area': 2000,
            'solar_panels': True,
            'green_roof': True,
            'rainwater_harvesting': False,
            'local_materials': True
        }

        analysis = await self.analyze_regenerative_potential(sample_building)
        print(f"   Overall Regenerative Score: {analysis['overall_regenerative_score']:.1%}")

        for category, score in analysis['category_scores'].items():
            print(f"   {category.title()}: {score:.1%}")

        print("\n💡 Top Recommendations:")
        for rec in analysis['recommendations'][:2]:
            print(f"   • {rec}")

        # Interactive AI chat demo
        print("\n💬 Conversational AI Demo:")
        sample_questions = [
            "How can I make my building net-positive energy?",
            "What water management strategies work best?",
            "How do I integrate ecosystems into my design?"
        ]

        for question in sample_questions:
            print(f"\n🧑 Question: {question}")
            response = await self.chat_with_ai(question)
            # Show first few lines of response
            lines = response.split('\n')[:4]
            for line in lines:
                print(f"🤖 {line}")
            print("   ... (truncated for demo)")

        print("\n" + "="*60)
        print("✨ RegeneraX Demo Complete!")
        print("This demonstrates how AI can guide regenerative urban design")
        print("to create cities that heal rather than harm our planet.")
        print("="*60)

async def main():
    """Main demo function"""
    demo = SimpleRegeneraX()
    await demo.run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")