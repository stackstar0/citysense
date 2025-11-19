"""
RegeneraX Conversational AI Interface
====================================

A sophisticated chatbot/IVR system that serves as the digital mind of the city,
capable of interpreting urban flows and providing regenerative architectural guidance.
This interface can answer complex questions about design decisions, resource optimization,
and ecological impact.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import sqlite3

# Import our core systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.city_brain import CityBrain
from core.data_processor import DataProcessor
from ai_engine.prediction_engine import PredictionEngine
from ai_engine.pattern_recognition import PatternRecognition
from ecosystem.impact_analyzer import ImpactAnalyzer
from regenerative.optimization_engine import OptimizationEngine

@dataclass
class ConversationContext:
    """Maintains context for ongoing conversations"""
    user_id: str
    session_id: str
    context_history: List[Dict[str, Any]]
    current_topic: str
    expertise_level: str  # 'citizen', 'planner', 'architect', 'expert'
    preferences: Dict[str, Any]

class RegenerativeArchitecturalAdvisor:
    """
    Advanced AI advisor that provides regenerative architectural guidance
    based on real-time city data and predictive modeling.
    """

    def __init__(self):
        self.city_brain = None
        self.data_processor = None
        self.prediction_engine = None
        self.pattern_recognition = None
        self.impact_analyzer = None
        self.optimization_engine = None

        # Knowledge base for architectural guidance
        self.architectural_principles = {
            'energy_efficiency': {
                'passive_solar': 'Optimize building orientation and fenestration for natural heating/cooling',
                'thermal_mass': 'Use materials that store and release thermal energy effectively',
                'insulation': 'Minimize heat transfer through building envelope',
                'natural_ventilation': 'Design for cross-ventilation and stack effect cooling'
            },
            'water_management': {
                'rainwater_harvesting': 'Collect and store precipitation for reuse',
                'greywater_systems': 'Treat and reuse water from sinks, showers, laundry',
                'permeable_surfaces': 'Allow water infiltration to reduce runoff',
                'bioswales': 'Natural drainage systems that filter stormwater'
            },
            'material_optimization': {
                'local_materials': 'Reduce transportation impacts with regional resources',
                'recycled_content': 'Incorporate waste streams into building materials',
                'durability': 'Design for longevity to minimize replacement cycles',
                'disassembly': 'Enable future material recovery and reuse'
            },
            'ecosystem_integration': {
                'biodiversity': 'Create habitats for local flora and fauna',
                'carbon_sequestration': 'Integrate plants and soils that capture CO2',
                'microclimate': 'Moderate temperature, humidity, and air quality',
                'food_production': 'Enable urban agriculture and composting'
            }
        }

        # Conversation patterns and intents
        self.intent_patterns = {
            'energy_question': [
                r'energy', r'power', r'electricity', r'heating', r'cooling', r'HVAC',
                r'solar', r'renewable', r'efficiency', r'consumption'
            ],
            'water_question': [
                r'water', r'rain', r'drainage', r'stormwater', r'irrigation', r'runoff',
                r'flood', r'drought', r'conservation', r'harvesting'
            ],
            'material_question': [
                r'material', r'concrete', r'steel', r'wood', r'recycl', r'waste',
                r'construction', r'durability', r'lifecycle', r'embodied'
            ],
            'ecosystem_question': [
                r'ecosystem', r'biodiversity', r'habitat', r'carbon', r'air quality',
                r'vegetation', r'green', r'natural', r'environment', r'climate'
            ],
            'cost_question': [
                r'cost', r'price', r'budget', r'economic', r'ROI', r'payback',
                r'investment', r'savings', r'value', r'affordable'
            ],
            'design_question': [
                r'design', r'architect', r'planning', r'layout', r'space',
                r'building', r'structure', r'form', r'function', r'aesthetic'
            ]
        }

        self.setup_logging()

    def setup_logging(self):
        """Setup logging for conversation tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize all core systems"""
        try:
            self.city_brain = CityBrain()
            self.data_processor = DataProcessor()
            self.prediction_engine = PredictionEngine()
            self.pattern_recognition = PatternRecognition()
            self.impact_analyzer = ImpactAnalyzer()
            self.optimization_engine = OptimizationEngine()

            await self.city_brain.initialize()
            await self.data_processor.initialize()
            await self.prediction_engine.initialize()
            await self.pattern_recognition.initialize()
            await self.impact_analyzer.initialize()
            await self.optimization_engine.initialize()

            self.logger.info("RegenerativeArchitecturalAdvisor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize advisor: {e}")
            raise

    def identify_intent(self, user_input: str) -> List[str]:
        """Identify the user's intent from their input"""
        user_input_lower = user_input.lower()
        identified_intents = []

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    identified_intents.append(intent)
                    break

        return identified_intents or ['general_question']

    async def get_current_city_data(self) -> Dict[str, Any]:
        """Retrieve current city vital signs and sensor data"""
        try:
            vital_signs = await self.city_brain.get_vital_signs()
            recent_data = await self.data_processor.get_recent_data(hours=1)

            return {
                'vital_signs': vital_signs,
                'sensor_data': recent_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error retrieving city data: {e}")
            return {}

    async def get_predictive_insights(self, domain: str) -> Dict[str, Any]:
        """Get predictive insights for specific domain"""
        try:
            if domain == 'energy':
                predictions = await self.prediction_engine.predict_energy_demand(hours_ahead=24)
            elif domain == 'water':
                predictions = await self.prediction_engine.predict_water_usage(hours_ahead=24)
            elif domain == 'air_quality':
                predictions = await self.prediction_engine.predict_air_quality(hours_ahead=24)
            else:
                predictions = await self.prediction_engine.predict_comprehensive()

            return predictions
        except Exception as e:
            self.logger.error(f"Error getting predictions for {domain}: {e}")
            return {}

    async def analyze_design_impact(self, design_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the potential impact of a design proposal"""
        try:
            impact_analysis = await self.impact_analyzer.analyze_impact(
                trigger_type='design_proposal',
                data=design_proposal
            )

            return impact_analysis
        except Exception as e:
            self.logger.error(f"Error analyzing design impact: {e}")
            return {}

    async def generate_regenerative_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """Generate specific regenerative design recommendations"""
        try:
            optimization_results = await self.optimization_engine.optimize(context)

            recommendations = []
            for action in optimization_results.get('recommended_actions', []):
                if action['category'] in ['building_design', 'site_planning', 'material_selection']:
                    recommendations.append(action['description'])

            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["I'm having trouble accessing the optimization engine right now."]

    def format_architectural_guidance(self, intents: List[str], city_data: Dict[str, Any]) -> str:
        """Format architectural guidance based on identified intents and current city data"""
        guidance_parts = []

        for intent in intents:
            if intent == 'energy_question':
                energy_data = city_data.get('vital_signs', {}).get('energy_efficiency', 0)
                guidance_parts.append(f"""
**Energy Efficiency Guidance:**
Current city energy efficiency: {energy_data:.1%}

Key strategies for regenerative energy design:
• **Passive Solar Design**: Orient buildings to maximize winter sun exposure and minimize summer heat gain
• **Thermal Mass**: Use materials like concrete, stone, or phase-change materials to moderate temperature swings
• **Natural Ventilation**: Design for cross-ventilation and stack effect to reduce mechanical cooling needs
• **Renewable Integration**: Consider on-site solar, wind, or geothermal systems
""")

            elif intent == 'water_question':
                water_data = city_data.get('vital_signs', {}).get('water_efficiency', 0)
                guidance_parts.append(f"""
**Water Management Guidance:**
Current city water efficiency: {water_data:.1%}

Regenerative water strategies:
• **Rainwater Harvesting**: Size cisterns based on roof area and local precipitation patterns
• **Greywater Systems**: Treat and reuse water from sinks, showers, and laundry
• **Permeable Surfaces**: Use porous pavement and green infrastructure to manage stormwater
• **Bioswales & Rain Gardens**: Natural systems that filter runoff while creating habitat
""")

            elif intent == 'material_question':
                guidance_parts.append(f"""
**Material Optimization Guidance:**

Regenerative material strategies:
• **Local Sourcing**: Use materials from within 500 miles to reduce transportation impacts
• **Recycled Content**: Incorporate post-consumer and post-industrial waste streams
• **Durability Focus**: Design for 100+ year lifespan to minimize replacement cycles
• **End-of-Life Planning**: Enable disassembly and material recovery for future use
• **Biogenic Materials**: Consider rapidly renewable resources like bamboo, hemp, or mycelium
""")

            elif intent == 'ecosystem_question':
                biodiversity_data = city_data.get('vital_signs', {}).get('biodiversity_index', 0)
                guidance_parts.append(f"""
**Ecosystem Integration Guidance:**
Current city biodiversity index: {biodiversity_data:.2f}

Regenerative ecosystem strategies:
• **Native Plant Integration**: Use local flora to support existing food webs
• **Habitat Creation**: Design green roofs, walls, and corridors for wildlife movement
• **Carbon Sequestration**: Maximize soil and plant-based carbon storage
• **Microclimate Design**: Use vegetation and water features to moderate temperature and humidity
• **Food Production**: Integrate edible landscapes and composting systems
""")

        return "\n".join(guidance_parts) if guidance_parts else self.get_general_guidance()

    def get_general_guidance(self) -> str:
        """Provide general regenerative architecture guidance"""
        return """
**Regenerative Architecture Principles:**

The goal of regenerative design is to create buildings and communities that give back more than they take from natural systems. Here are core principles:

**🌱 Net-Positive Impact**: Design to produce more energy, clean more water, and sequester more carbon than consumed

**🔄 Circular Systems**: Create closed-loop systems where waste from one process becomes input for another

**🏞️ Ecosystem Integration**: Work with natural systems rather than against them

**⚖️ Cost-Ecology Balance**: Recognize that long-term economic value comes from ecological health

**🔬 Data-Driven Decisions**: Use real-time monitoring and predictive modeling to optimize performance

**🌍 Resilience Focus**: Design for adaptation to changing climate and social conditions

Would you like specific guidance on any of these areas?
"""

    async def process_conversation(self, user_input: str, context: ConversationContext) -> str:
        """Process a conversation turn and generate response"""
        try:
            # Identify user intent
            intents = self.identify_intent(user_input)

            # Get current city data
            city_data = await self.get_current_city_data()

            # Generate base architectural guidance
            guidance = self.format_architectural_guidance(intents, city_data)

            # Add predictive insights if relevant
            if any(intent in intents for intent in ['energy_question', 'water_question']):
                for intent in intents:
                    domain = intent.replace('_question', '')
                    if domain in ['energy', 'water']:
                        predictions = await self.get_predictive_insights(domain)
                        if predictions:
                            guidance += f"\n\n**Predictive Insights for {domain.title()}:**\n"
                            guidance += f"Expected trends over next 24 hours based on current patterns and weather forecasts.\n"

            # Add specific recommendations if this is a design question
            if 'design_question' in intents:
                recommendations = await self.generate_regenerative_recommendations({
                    'user_input': user_input,
                    'city_data': city_data,
                    'context': context.context_history
                })

                if recommendations:
                    guidance += "\n\n**Specific Recommendations:**\n"
                    for i, rec in enumerate(recommendations[:5], 1):
                        guidance += f"{i}. {rec}\n"

            # Update conversation context
            context.context_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'intents': intents,
                'response_length': len(guidance)
            })

            # Keep context history manageable
            if len(context.context_history) > 10:
                context.context_history = context.context_history[-10:]

            return guidance

        except Exception as e:
            self.logger.error(f"Error processing conversation: {e}")
            return f"I'm experiencing some technical difficulties right now. Please try rephrasing your question about regenerative architecture and design."

class ConversationalInterface:
    """
    Main interface for chatbot/IVR interactions
    """

    def __init__(self):
        self.advisor = RegenerativeArchitecturalAdvisor()
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the conversational interface"""
        await self.advisor.initialize()
        self.logger.info("ConversationalInterface initialized")

    def create_session(self, user_id: str, expertise_level: str = 'citizen') -> str:
        """Create a new conversation session"""
        import uuid
        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            context_history=[],
            current_topic='general',
            expertise_level=expertise_level,
            preferences={}
        )

        return session_id

    async def chat(self, session_id: str, user_input: str) -> str:
        """Process a chat message and return response"""
        if session_id not in self.active_sessions:
            return "Session not found. Please start a new conversation."

        context = self.active_sessions[session_id]
        response = await self.advisor.process_conversation(user_input, context)

        return response

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        context = self.active_sessions[session_id]
        return {
            'session_id': session_id,
            'user_id': context.user_id,
            'expertise_level': context.expertise_level,
            'conversation_turns': len(context.context_history),
            'topics_discussed': list(set([turn.get('intents', []) for turn in context.context_history])),
            'current_topic': context.current_topic
        }

# Example usage and testing
async def main():
    """Example usage of the conversational AI interface"""
    interface = ConversationalInterface()
    await interface.initialize()

    # Create a session
    session_id = interface.create_session('user123', 'architect')

    # Example conversations
    test_questions = [
        "How can I design a building that actually improves the local ecosystem?",
        "What's the most cost-effective way to achieve net-positive energy in my design?",
        "How should current weather patterns influence my material choices?",
        "Can you analyze the environmental impact of using reclaimed materials?",
        "What regenerative design strategies work best in urban environments?"
    ]

    print("=== RegeneraX Conversational AI Demo ===\n")

    for question in test_questions:
        print(f"🧑 User: {question}")
        print("🤖 RegeneraX:", end=" ")

        response = await interface.chat(session_id, question)
        print(f"{response}\n")
        print("-" * 80 + "\n")

    # Show session summary
    summary = await interface.get_session_summary(session_id)
    print("Session Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())