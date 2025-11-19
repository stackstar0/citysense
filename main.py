"""
RegeneraX - Regenerative Urban Intelligence Platform
=================================================

Main entry point for the RegeneraX platform - a revolutionary AI-powered system
that perceives cities as living ecosystems and guides regenerative architectural decisions.

This system transforms urban planning from a responsible compromise into a
regenerative catalyst, using advanced AI to decode how structures can reduce
stress on natural systems, extend lifecycles, optimize resources, and generate
net-positive environmental impact.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Core system imports
from core.city_brain import CityBrain
from core.sensor_manager import SensorManager
from core.data_processor import DataProcessor

# AI engine imports
from ai_engine.prediction_engine import PredictionEngine
from ai_engine.pattern_recognition import PatternRecognition

# Ecosystem and optimization imports
from ecosystem.impact_analyzer import ImpactAnalyzer
from regenerative.optimization_engine import OptimizationEngine

# Interface imports
from interfaces.conversational_ai import ConversationalInterface
from interfaces.vr_interface import RegenerativeVREngine

# Visualization and API imports
from visualization.websocket_manager import WebSocketManager
from api.routes import create_app

# Setup comprehensive logging
def setup_logging():
    """Setup comprehensive logging for the RegeneraX system"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "regenerax.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)

class RegeneraXPlatform:
    """
    Main platform orchestrator for RegeneraX

    Coordinates all system components to provide comprehensive urban intelligence:
    - Real-time sensing and monitoring
    - AI-powered analysis and prediction
    - Regenerative optimization
    - Immersive interfaces (VR/AR, Conversational AI)
    - Real-time visualization and API access
    """

    def __init__(self):
        self.logger = setup_logging()
        self.components = {}
        self.interfaces = {}
        self.running = False
        self.start_time = None

        # System configuration
        self.config = {
            'api_host': os.getenv('API_HOST', '0.0.0.0'),
            'api_port': int(os.getenv('API_PORT', 8000)),
            'websocket_port': int(os.getenv('WEBSOCKET_PORT', 8080)),
            'city_name': os.getenv('CITY_NAME', 'RegeneraX City'),
            'enable_vr': os.getenv('ENABLE_VR_INTERFACE', 'true').lower() == 'true',
            'enable_ai_chat': os.getenv('ENABLE_AI_CHAT', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }

    async def initialize_core_components(self):
        """Initialize core urban intelligence components"""
        self.logger.info("🏗️  Initializing core urban intelligence components...")

        # Core sensing and processing
        self.components['city_brain'] = CityBrain()
        self.components['sensor_manager'] = SensorManager()
        self.components['data_processor'] = DataProcessor()

        # Initialize core components
        for name in ['city_brain', 'sensor_manager', 'data_processor']:
            self.logger.info(f"   Initializing {name}...")
            await self.components[name].initialize()

        self.logger.info("✅ Core components initialized")

    async def initialize_ai_engine(self):
        """Initialize AI and machine learning components"""
        self.logger.info("🧠 Initializing AI engine components...")

        # AI and ML components
        self.components['prediction_engine'] = PredictionEngine()
        self.components['pattern_recognition'] = PatternRecognition()

        # Initialize AI components
        for name in ['prediction_engine', 'pattern_recognition']:
            self.logger.info(f"   Initializing {name}...")
            await self.components[name].initialize()

        self.logger.info("✅ AI engine initialized")

    async def initialize_ecosystem_components(self):
        """Initialize ecosystem modeling and optimization"""
        self.logger.info("🌍 Initializing ecosystem modeling components...")

        # Ecosystem and optimization
        self.components['impact_analyzer'] = ImpactAnalyzer()
        self.components['optimization_engine'] = OptimizationEngine()

        # Initialize ecosystem components
        for name in ['impact_analyzer', 'optimization_engine']:
            self.logger.info(f"   Initializing {name}...")
            await self.components[name].initialize()

        self.logger.info("✅ Ecosystem modeling initialized")

    async def initialize_interfaces(self):
        """Initialize user interfaces"""
        self.logger.info("🎮 Initializing user interfaces...")

        # Conversational AI interface
        if self.config['enable_ai_chat']:
            self.logger.info("   Initializing conversational AI interface...")
            self.interfaces['conversational_ai'] = ConversationalInterface()
            await self.interfaces['conversational_ai'].initialize()

        # VR/AR interface
        if self.config['enable_vr']:
            self.logger.info("   Initializing VR/AR interface...")
            self.interfaces['vr_engine'] = RegenerativeVREngine()
            await self.interfaces['vr_engine'].initialize()

        # WebSocket manager for real-time communication
        self.logger.info("   Initializing WebSocket manager...")
        self.components['websocket_manager'] = WebSocketManager()
        await self.components['websocket_manager'].initialize()

        self.logger.info("✅ User interfaces initialized")

    async def initialize(self):
        """Initialize the complete RegeneraX platform"""
        try:
            self.logger.info("🌱 Initializing RegeneraX - Regenerative Urban Intelligence Platform")
            self.logger.info("=" * 80)

            # Initialize components in order
            await self.initialize_core_components()
            await self.initialize_ai_engine()
            await self.initialize_ecosystem_components()
            await self.initialize_interfaces()

            self.running = True
            self.start_time = datetime.now()

            self.logger.info("=" * 80)
            self.logger.info("🚀 RegeneraX platform initialized successfully!")
            self.logger.info(f"📍 City: {self.config['city_name']}")
            self.logger.info(f"⚙️  Core Components: {len(self.components)}")
            self.logger.info(f"🎯 Interfaces: {len(self.interfaces)}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RegeneraX platform: {e}")
            raise

    async def start_services(self):
        """Start all RegeneraX services"""
        if not self.running:
            await self.initialize()

        self.logger.info("🚀 Starting RegeneraX services...")

        # Start core services
        self.logger.info("   Starting city brain orchestrator...")
        await self.components['city_brain'].start()

        self.logger.info("   Starting sensor monitoring...")
        await self.components['sensor_manager'].start()

        # Start WebSocket server for real-time communication
        self.logger.info("   Starting WebSocket server...")
        websocket_task = asyncio.create_task(
            self.components['websocket_manager'].start(port=self.config['websocket_port'])
        )

        # Start web API server
        self.logger.info("   Starting web API server...")
        app = create_app()

        # Add CORS for VR and web applications
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        import uvicorn
        config = uvicorn.Config(
            app,
            host=self.config['api_host'],
            port=self.config['api_port'],
            log_level=self.config['log_level'].lower()
        )
        server = uvicorn.Server(config)
        api_task = asyncio.create_task(server.serve())

        # Print startup information
        self.logger.info("=" * 80)
        self.logger.info("🌟 RegeneraX is now LIVE!")
        self.logger.info("=" * 80)
        self.logger.info(f"🌐 Main Dashboard: http://localhost:{self.config['api_port']}")
        self.logger.info(f"📖 API Documentation: http://localhost:{self.config['api_port']}/docs")
        self.logger.info(f"🔌 WebSocket Endpoint: ws://localhost:{self.config['websocket_port']}")

        if self.config['enable_vr']:
            self.logger.info(f"🥽 VR Interface: http://localhost:{self.config['api_port']}/vr")

        if self.config['enable_ai_chat']:
            self.logger.info(f"💬 Conversational AI: Available via API and WebSocket")

        self.logger.info("=" * 80)
        self.logger.info("🎯 Ready to transform cities into regenerative ecosystems!")
        self.logger.info("Press Ctrl+C to gracefully shutdown")

        # Wait for services
        await asyncio.gather(websocket_task, api_task, return_exceptions=True)

    async def stop(self):
        """Gracefully stop the RegeneraX platform"""
        self.logger.info("🛑 Stopping RegeneraX platform...")

        # Stop all components
        for name, component in self.components.items():
            if hasattr(component, 'stop'):
                self.logger.info(f"   Stopping {name}...")
                try:
                    await component.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")

        # Stop interfaces
        for name, interface in self.interfaces.items():
            if hasattr(interface, 'stop'):
                self.logger.info(f"   Stopping {name}...")
                try:
                    await interface.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")

        self.running = False

        if self.start_time:
            uptime = datetime.now() - self.start_time
            self.logger.info(f"⏱️  Total uptime: {uptime}")

        self.logger.info("👋 RegeneraX platform stopped gracefully")

    def print_banner(self):
        """Print the RegeneraX startup banner"""
        banner = """
        ╭─────────────────────────────────────────────────────────────────╮
        │                                                                 │
        │    🌱 RegeneraX - Regenerative Urban Intelligence Platform      │
        │                                                                 │
        │    Transforming cities into living, regenerative ecosystems     │
        │    through AI-powered design intelligence and real-time         │
        │    environmental monitoring.                                    │
        │                                                                 │
        │    "Where cost and ecology unite, design intelligence thrives,  │
        │     and cities become regenerative catalysts for planetary      │
        │     health."                                                    │
        │                                                                 │
        ╰─────────────────────────────────────────────────────────────────╯
        """
        print(banner)

async def run_demo():
    """Run a quick demo of RegeneraX capabilities"""
    logger = logging.getLogger(__name__)

    print("\n🎯 Running RegeneraX Demo...")
    print("=" * 50)

    # Create platform instance
    platform = RegeneraXPlatform()
    await platform.initialize()

    try:
        # Demo conversational AI
        if 'conversational_ai' in platform.interfaces:
            print("\n💬 Testing Conversational AI Interface:")
            interface = platform.interfaces['conversational_ai']
            session_id = interface.create_session('demo_user', 'architect')

            response = await interface.chat(
                session_id,
                "How can I design a building that actually improves the local ecosystem?"
            )
            print("🤖 AI Response:", response[:200] + "..." if len(response) > 200 else response)

        # Demo VR capabilities
        if 'vr_engine' in platform.interfaces:
            print("\n🥽 Testing VR Interface:")
            vr_engine = platform.interfaces['vr_engine']
            session_id = await vr_engine.create_vr_session('demo_user', 'webxr', 'design')
            print(f"✅ VR session created: {session_id}")

        # Demo city intelligence
        print("\n🏙️ Testing City Intelligence:")
        city_brain = platform.components['city_brain']
        vital_signs = await city_brain.get_vital_signs()
        print(f"🔍 City Health Score: {vital_signs.get('overall_health', 0):.2%}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo error: {e}")

    finally:
        await platform.stop()

async def main():
    """Main entry point for RegeneraX platform"""
    platform = RegeneraXPlatform()
    platform.print_banner()

    # Check for demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        await run_demo()
        return

    try:
        await platform.start_services()
    except KeyboardInterrupt:
        platform.logger.info("\n🛑 Received shutdown signal...")
        await platform.stop()
    except Exception as e:
        platform.logger.error(f"❌ System error: {e}")
        await platform.stop()
        sys.exit(1)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            print("""
RegeneraX - Regenerative Urban Intelligence Platform

Usage:
  python main.py          # Start full platform
  python main.py demo     # Run demo mode
  python main.py help     # Show this help

Environment Variables:
  CITY_NAME               # Name of the city (default: "RegeneraX City")
  API_HOST                # API host (default: "0.0.0.0")
  API_PORT                # API port (default: 8000)
  WEBSOCKET_PORT          # WebSocket port (default: 8080)
  ENABLE_VR_INTERFACE     # Enable VR features (default: true)
  ENABLE_AI_CHAT          # Enable conversational AI (default: true)
  LOG_LEVEL               # Logging level (default: INFO)

For more information, visit: https://docs.regenerax.org
            """)
            sys.exit(0)

    # Run the platform
    asyncio.run(main())