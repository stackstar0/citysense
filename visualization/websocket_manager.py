"""
WebSocket Manager - Real-time data streaming for city intelligence dashboard
"""

import asyncio
import logging
import json
from typing import Set, Dict, Any, List
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections for real-time city data streaming
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now(),
            "subscriptions": ["all"]  # Default subscription
        }

        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

        # Send welcome message
        await self.send_to_websocket(websocket, {
            "type": "connection_established",
            "message": "Connected to RegeneraX City Intelligence",
            "timestamp": datetime.now().isoformat(),
            "available_channels": [
                "vital_signs", "sensor_data", "insights", "optimizations",
                "patterns", "impacts", "system_status"
            ]
        })

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_to_websocket(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Error sending data to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, data: Dict[str, Any], channel: str = "all"):
        """Broadcast data to all connected WebSocket clients"""
        if not self.active_connections:
            return

        # Add metadata
        broadcast_data = {
            **data,
            "channel": channel,
            "broadcast_time": datetime.now().isoformat(),
            "connection_count": len(self.active_connections)
        }

        # Send to all connections
        disconnected = []
        for websocket in self.active_connections:
            try:
                # Check if connection is subscribed to this channel
                subscriptions = self.connection_info.get(websocket, {}).get("subscriptions", ["all"])
                if channel in subscriptions or "all" in subscriptions:
                    await websocket.send_text(json.dumps(broadcast_data, default=str))
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)

    async def send_vital_signs_update(self, vital_signs):
        """Send city vital signs update"""
        await self.broadcast({
            "type": "vital_signs_update",
            "data": vital_signs.to_dict() if vital_signs else {}
        }, "vital_signs")

    async def send_sensor_data_update(self, sensor_data: Dict[str, Any]):
        """Send sensor data update"""
        await self.broadcast({
            "type": "sensor_data_update",
            "data": sensor_data
        }, "sensor_data")

    async def send_insights_update(self, insights: List[Dict[str, Any]]):
        """Send insights update"""
        await self.broadcast({
            "type": "insights_update",
            "data": insights
        }, "insights")

    async def send_optimization_update(self, optimizations: List[Dict[str, Any]]):
        """Send optimization actions update"""
        await self.broadcast({
            "type": "optimization_update",
            "data": optimizations
        }, "optimizations")

    async def send_pattern_update(self, patterns: List[Dict[str, Any]]):
        """Send pattern recognition update"""
        await self.broadcast({
            "type": "pattern_update",
            "data": patterns
        }, "patterns")

    async def send_impact_update(self, impact_assessment: Dict[str, Any]):
        """Send impact assessment update"""
        await self.broadcast({
            "type": "impact_update",
            "data": impact_assessment
        }, "impacts")

    async def send_system_status_update(self, status: Dict[str, Any]):
        """Send system status update"""
        await self.broadcast({
            "type": "system_status_update",
            "data": status
        }, "system_status")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "connection_details": [
                {
                    "connected_at": info["connected_at"].isoformat(),
                    "subscriptions": info["subscriptions"]
                }
                for info in self.connection_info.values()
            ]
        }