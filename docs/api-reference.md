# RegeneraX API Reference

Complete documentation for RegeneraX REST API endpoints, WebSocket connections, and integration patterns.

## Base URL

```
Production: https://api.regenerax.org/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

RegeneraX uses API keys for authentication. Include your API key in the header:

```bash
curl -H "X-API-Key: your_api_key_here" https://api.regenerax.org/api/v1/health
```

## Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "timestamp": "2024-01-20T10:30:00Z",
  "request_id": "req_123456789"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter: city_name",
    "details": {
      "field": "city_name",
      "expected": "string",
      "received": "null"
    }
  },
  "timestamp": "2024-01-20T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Endpoints

### System Health

#### GET /health
Check system health and component status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600,
    "components": {
      "database": "healthy",
      "sensors": "healthy",
      "ai_engine": "healthy",
      "websocket": "healthy"
    },
    "performance": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "active_connections": 23
    }
  }
}
```

### City Vital Signs

#### GET /vital-signs
Get current city vital signs and health metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_health": 0.78,
    "air_quality": 0.72,
    "energy_efficiency": 0.83,
    "water_efficiency": 0.71,
    "carbon_footprint": 0.65,
    "biodiversity_index": 0.69,
    "economic_vitality": 0.85,
    "social_equity": 0.74,
    "last_updated": "2024-01-20T10:25:00Z",
    "trend": {
      "direction": "improving",
      "change_rate": 0.02
    }
  }
}
```

#### GET /vital-signs/history
Get historical vital signs data.

**Parameters:**
- `days` (integer): Number of days of history (default: 7, max: 365)
- `interval` (string): Data interval - "hour", "day", "week" (default: "hour")

**Response:**
```json
{
  "success": true,
  "data": {
    "timeframe": {
      "start": "2024-01-13T10:30:00Z",
      "end": "2024-01-20T10:30:00Z",
      "interval": "hour"
    },
    "metrics": [
      {
        "timestamp": "2024-01-20T09:00:00Z",
        "overall_health": 0.76,
        "air_quality": 0.71,
        "energy_efficiency": 0.82,
        // ... other metrics
      }
      // ... more data points
    ]
  }
}
```

### Sensor Data

#### GET /sensors
List all available sensors and their current status.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_sensors": 45,
    "active_sensors": 42,
    "categories": {
      "air_quality": 12,
      "energy": 8,
      "water": 6,
      "noise": 10,
      "traffic": 7,
      "biodiversity": 4
    },
    "sensors": [
      {
        "sensor_id": "air_001",
        "type": "air_quality",
        "location": {
          "latitude": 37.7749,
          "longitude": -122.4194,
          "address": "Downtown Sensor Station"
        },
        "status": "active",
        "last_reading": "2024-01-20T10:28:00Z",
        "battery_level": 87
      }
      // ... more sensors
    ]
  }
}
```

#### GET /sensors/{sensor_id}/data
Get data from a specific sensor.

**Parameters:**
- `hours` (integer): Hours of data to retrieve (default: 24, max: 168)
- `limit` (integer): Maximum number of readings (default: 100, max: 1000)

**Response:**
```json
{
  "success": true,
  "data": {
    "sensor_id": "air_001",
    "sensor_type": "air_quality",
    "readings": [
      {
        "timestamp": "2024-01-20T10:25:00Z",
        "pm25": 12.5,
        "pm10": 18.3,
        "co2": 412,
        "no2": 23.1,
        "o3": 45.6,
        "quality_index": 0.72
      }
      // ... more readings
    ],
    "statistics": {
      "average": 0.71,
      "min": 0.45,
      "max": 0.89,
      "trend": "stable"
    }
  }
}
```

### AI Engine

#### GET /insights
Get AI-generated insights about city conditions.

**Parameters:**
- `category` (string): Focus category - "energy", "water", "air", "biodiversity", "all" (default: "all")
- `timeframe` (string): Analysis timeframe - "day", "week", "month" (default: "week")

**Response:**
```json
{
  "success": true,
  "data": {
    "insights": [
      {
        "id": "insight_001",
        "category": "energy",
        "title": "Peak Energy Efficiency Opportunity",
        "description": "Building retrofits in the downtown district could reduce energy consumption by 15% based on current usage patterns.",
        "confidence": 0.87,
        "impact_potential": "high",
        "recommended_actions": [
          "Install smart HVAC systems in buildings over 20 years old",
          "Implement dynamic lighting controls",
          "Add solar panels to suitable rooftops"
        ],
        "created_at": "2024-01-20T10:15:00Z"
      }
      // ... more insights
    ],
    "summary": {
      "total_insights": 12,
      "high_impact": 4,
      "medium_impact": 6,
      "low_impact": 2
    }
  }
}
```

#### GET /predictions
Get AI predictions for various city metrics.

**Parameters:**
- `metric` (string): Metric to predict - "energy", "water", "air_quality", "traffic" (required)
- `hours_ahead` (integer): Hours to predict ahead (default: 24, max: 168)

**Response:**
```json
{
  "success": true,
  "data": {
    "metric": "energy",
    "prediction_horizon": 24,
    "predictions": [
      {
        "timestamp": "2024-01-20T11:00:00Z",
        "predicted_value": 1250.5,
        "confidence_interval": {
          "lower": 1180.2,
          "upper": 1320.8
        },
        "confidence": 0.92
      }
      // ... more predictions
    ],
    "model_info": {
      "model_type": "ensemble",
      "last_trained": "2024-01-19T02:00:00Z",
      "accuracy": 0.94
    }
  }
}
```

#### GET /patterns
Get detected patterns in city data.

**Parameters:**
- `type` (string): Pattern type - "temporal", "spatial", "anomaly", "all" (default: "all")
- `significance` (string): Minimum significance - "low", "medium", "high" (default: "medium")

**Response:**
```json
{
  "success": true,
  "data": {
    "patterns": [
      {
        "pattern_id": "pattern_001",
        "type": "temporal",
        "category": "energy",
        "description": "Weekly energy consumption peak occurs on Tuesdays at 2 PM",
        "significance": "high",
        "confidence": 0.91,
        "frequency": "weekly",
        "locations": ["downtown", "midtown"],
        "detected_at": "2024-01-20T08:00:00Z"
      }
      // ... more patterns
    ],
    "summary": {
      "total_patterns": 8,
      "by_type": {
        "temporal": 5,
        "spatial": 2,
        "anomaly": 1
      }
    }
  }
}
```

### Ecosystem Analysis

#### POST /impact/analyze
Analyze the impact of a proposed change or intervention.

**Request Body:**
```json
{
  "intervention_type": "building_retrofit",
  "parameters": {
    "building_count": 50,
    "area": "downtown",
    "efficiency_improvement": 0.25,
    "renewable_energy": true
  },
  "analysis_scope": ["energy", "carbon", "economic"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "intervention_id": "analysis_001",
    "estimated_impact": {
      "energy": {
        "annual_savings_kwh": 2500000,
        "cost_savings_usd": 375000,
        "carbon_reduction_kg": 1875000
      },
      "economic": {
        "initial_investment_usd": 5000000,
        "payback_period_years": 8.2,
        "roi_10_year": 1.85
      },
      "environmental": {
        "carbon_footprint_reduction": 0.15,
        "air_quality_improvement": 0.08,
        "biodiversity_impact": 0.02
      }
    },
    "confidence": 0.83,
    "assumptions": [
      "Current energy prices remain stable",
      "Building occupancy rates maintain current levels",
      "Technology performance meets manufacturer specifications"
    ]
  }
}
```

### Optimization

#### POST /optimization/optimize
Request optimization recommendations for city systems.

**Request Body:**
```json
{
  "context": {
    "area": "downtown",
    "focus": "energy_efficiency",
    "budget": 1000000,
    "timeframe": "1_year"
  },
  "constraints": {
    "environmental_impact": "minimize",
    "social_disruption": "low",
    "technology_readiness": "high"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_001",
    "recommended_actions": [
      {
        "action_id": "action_001",
        "category": "building_efficiency",
        "description": "Retrofit HVAC systems in 25 commercial buildings",
        "priority": "high",
        "estimated_cost": 400000,
        "expected_impact": {
          "energy_savings": 0.18,
          "carbon_reduction": 0.12,
          "payback_years": 6.5
        },
        "implementation_timeline": "6_months"
      }
      // ... more actions
    ],
    "overall_impact": {
      "total_cost": 950000,
      "energy_efficiency_improvement": 0.23,
      "carbon_reduction": 0.16,
      "roi_5_year": 1.45
    },
    "confidence": 0.89
  }
}
```

### Data Export

#### GET /data/export
Export city data in various formats.

**Parameters:**
- `format` (string): Export format - "csv", "json", "parquet" (default: "json")
- `data_types` (array): Data types to include - ["sensors", "vital_signs", "insights", "patterns"]
- `start_date` (string): Start date (ISO 8601 format)
- `end_date` (string): End date (ISO 8601 format)
- `compression` (string): Compression - "none", "gzip", "zip" (default: "none")

**Response:**
For JSON format:
```json
{
  "success": true,
  "data": {
    "export_id": "export_001",
    "download_url": "https://api.regenerax.org/downloads/export_001.json.gz",
    "expires_at": "2024-01-21T10:30:00Z",
    "file_size": 2048576,
    "record_count": 15000
  }
}
```

### Configuration

#### GET /config
Get current system configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "city_name": "San Francisco",
    "timezone": "America/Los_Angeles",
    "sensor_update_interval": 30,
    "ai_prediction_interval": 300,
    "features": {
      "vr_interface": true,
      "conversational_ai": true,
      "real_time_alerts": true
    },
    "limits": {
      "api_rate_limit": 1000,
      "max_export_records": 100000,
      "max_prediction_horizon": 168
    }
  }
}
```

#### PUT /config
Update system configuration (admin only).

**Request Body:**
```json
{
  "sensor_update_interval": 60,
  "ai_prediction_interval": 600,
  "features": {
    "real_time_alerts": false
  }
}
```

### Analytics

#### GET /analytics/summary
Get comprehensive analytics summary.

**Parameters:**
- `period` (string): Analysis period - "day", "week", "month", "year" (default: "week")

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "week",
    "timeframe": {
      "start": "2024-01-14T00:00:00Z",
      "end": "2024-01-21T00:00:00Z"
    },
    "metrics": {
      "total_sensor_readings": 45000,
      "ai_predictions_generated": 1200,
      "insights_created": 25,
      "patterns_detected": 8,
      "api_requests": 15000,
      "active_users": 150
    },
    "trends": {
      "air_quality": {
        "direction": "improving",
        "change_percent": 5.2
      },
      "energy_efficiency": {
        "direction": "stable",
        "change_percent": 0.8
      }
    },
    "top_insights": [
      {
        "title": "Peak Energy Efficiency Opportunity",
        "impact": "high",
        "confidence": 0.87
      }
    ]
  }
}
```

## WebSocket API

### Connection

Connect to real-time data streams:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = function() {
    console.log('Connected to RegeneraX WebSocket');

    // Subscribe to vital signs updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'vital_signs'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

### Available Channels

#### vital_signs
Real-time city vital signs updates.

**Subscription:**
```json
{
  "type": "subscribe",
  "channel": "vital_signs"
}
```

**Message Format:**
```json
{
  "channel": "vital_signs",
  "timestamp": "2024-01-20T10:30:00Z",
  "data": {
    "overall_health": 0.78,
    "air_quality": 0.72,
    "energy_efficiency": 0.83
    // ... other metrics
  }
}
```

#### sensors
Individual sensor data updates.

**Subscription:**
```json
{
  "type": "subscribe",
  "channel": "sensors",
  "filter": {
    "sensor_types": ["air_quality", "energy"],
    "locations": ["downtown"]
  }
}
```

#### insights
New AI insights as they're generated.

**Subscription:**
```json
{
  "type": "subscribe",
  "channel": "insights",
  "filter": {
    "categories": ["energy", "water"],
    "min_confidence": 0.8
  }
}
```

#### alerts
System alerts and anomalies.

**Subscription:**
```json
{
  "type": "subscribe",
  "channel": "alerts",
  "filter": {
    "severity": ["medium", "high"]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Invalid or missing API key |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| `INTERNAL_ERROR` | Server internal error |
| `SERVICE_UNAVAILABLE` | Dependent service unavailable |
| `DATA_PROCESSING_ERROR` | Error processing data |
| `MODEL_ERROR` | AI model error |

## Rate Limits

- **Standard Plan**: 1,000 requests per hour
- **Professional Plan**: 10,000 requests per hour
- **Enterprise Plan**: Custom limits

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642694400
```

## SDKs and Libraries

### Python

```python
import regenerax

client = regenerax.Client(api_key='your_api_key')

# Get vital signs
vital_signs = client.get_vital_signs()

# Get predictions
predictions = client.get_predictions('energy', hours_ahead=24)

# Analyze impact
impact = client.analyze_impact({
    'intervention_type': 'solar_installation',
    'parameters': {'capacity_kw': 1000}
})
```

### JavaScript

```javascript
import RegeneraX from 'regenerax-js';

const client = new RegeneraX('your_api_key');

// Get vital signs
const vitalSigns = await client.getVitalSigns();

// Subscribe to real-time updates
client.subscribe('vital_signs', (data) => {
    console.log('New vital signs:', data);
});
```

### cURL Examples

```bash
# Get vital signs
curl -H "X-API-Key: your_api_key" \
     https://api.regenerax.org/api/v1/vital-signs

# Get predictions
curl -H "X-API-Key: your_api_key" \
     "https://api.regenerax.org/api/v1/predictions?metric=energy&hours_ahead=24"

# Analyze impact
curl -X POST \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"intervention_type":"building_retrofit","parameters":{"building_count":10}}' \
     https://api.regenerax.org/api/v1/impact/analyze
```

## Changelog

### v1.0.0 (2024-01-20)
- Initial API release
- Core endpoints for vital signs, sensors, predictions
- WebSocket support for real-time data
- VR interface integration

### v0.9.0 (2024-01-15)
- Beta release
- AI insights and pattern detection
- Impact analysis capabilities
- Basic export functionality

For the complete API changelog, see [CHANGELOG.md](CHANGELOG.md).

## Support

- **Documentation**: https://docs.regenerax.org
- **GitHub Issues**: https://github.com/your-org/regenerax/issues
- **Email**: api-support@regenerax.org
- **Community**: https://discord.gg/regenerax