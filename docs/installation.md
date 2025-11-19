# RegeneraX Installation Guide

This guide provides comprehensive instructions for installing and configuring RegeneraX on various platforms and deployment scenarios.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for initial setup and data sources

### Recommended Requirements
- **OS**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.11
- **RAM**: 16GB for large city datasets
- **Storage**: 10GB+ for historical data storage
- **GPU**: NVIDIA GPU with CUDA support (optional, for ML acceleration)
- **Network**: Stable broadband for real-time data streams

## Installation Methods

### 1. Quick Start (Development)

The fastest way to get RegeneraX running for development and testing:

```bash
# Clone the repository
git clone https://github.com/your-org/regenerax.git
cd citysense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "
from core.data_processor import DataProcessor
import asyncio
async def init():
    dp = DataProcessor()
    await dp.initialize()
asyncio.run(init())
"

# Start the system
python main.py
```

### 2. Docker Installation (Recommended)

Docker provides the most consistent deployment across different environments:

```bash
# Clone the repository
git clone https://github.com/your-org/regenerax.git
cd citysense

# Build the Docker image
docker build -t regenerax:latest .

# Run with default settings
docker run -p 8000:8000 -p 8080:8080 regenerax:latest

# Run with custom configuration
docker run -p 8000:8000 -p 8080:8080 \
  -e CITY_NAME="San Francisco" \
  -e DATABASE_URL="sqlite:///data/city.db" \
  -e LOG_LEVEL="INFO" \
  -v $(pwd)/data:/app/data \
  regenerax:latest
```

### 3. Production Deployment

For production deployments with high availability and scalability:

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  regenerax-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/regenerax
      - REDIS_URL=redis://cache:6379
      - LOG_LEVEL=INFO
    depends_on:
      - db
      - cache
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  regenerax-websocket:
    build: .
    command: python -m visualization.websocket_manager
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/regenerax
      - REDIS_URL=redis://cache:6379
    depends_on:
      - db
      - cache
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: regenerax
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  cache:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - regenerax-api
      - regenerax-websocket
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: regenerax
spec:
  replicas: 3
  selector:
    matchLabels:
      app: regenerax
  template:
    metadata:
      labels:
        app: regenerax
    spec:
      containers:
      - name: regenerax
        image: regenerax:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: regenerax-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: regenerax-service
spec:
  selector:
    app: regenerax
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Configuration

### Environment Variables

RegeneraX can be configured using environment variables:

```bash
# Core Configuration
export CITY_NAME="Your City Name"
export CITY_TIMEZONE="America/New_York"
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Database Configuration
export DATABASE_URL="sqlite:///data/city.db"
# For PostgreSQL: postgresql://user:pass@localhost:5432/regenerax
# For MySQL: mysql://user:pass@localhost:3306/regenerax

# Cache Configuration (optional)
export REDIS_URL="redis://localhost:6379"

# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_WORKERS="4"

# WebSocket Configuration
export WEBSOCKET_HOST="0.0.0.0"
export WEBSOCKET_PORT="8080"

# Sensor Configuration
export ENABLE_MOCK_SENSORS="true"  # Use simulated sensors for testing
export SENSOR_UPDATE_INTERVAL="30"  # Seconds between sensor readings

# AI/ML Configuration
export ML_MODEL_PATH="/app/models"
export ENABLE_GPU_ACCELERATION="false"
export PREDICTION_BATCH_SIZE="100"

# VR Interface Configuration
export ENABLE_VR_INTERFACE="true"
export WEBXR_ENABLED="true"

# External Data Sources
export WEATHER_API_KEY="your_api_key"
export TRAFFIC_API_KEY="your_api_key"
export AIR_QUALITY_API_KEY="your_api_key"
```

### Configuration File

Alternatively, create a `config.yaml` file:

```yaml
# config.yaml
city:
  name: "Your City Name"
  timezone: "America/New_York"
  coordinates:
    latitude: 37.7749
    longitude: -122.4194

database:
  url: "sqlite:///data/city.db"
  echo_queries: false
  pool_size: 10
  max_overflow: 20

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins:
    - "http://localhost:3000"
    - "https://yourdomain.com"

websocket:
  host: "0.0.0.0"
  port: 8080
  max_connections: 1000

sensors:
  mock_enabled: true
  update_interval: 30
  categories:
    - air_quality
    - energy
    - water
    - noise
    - traffic
    - biodiversity

ml:
  model_path: "/app/models"
  gpu_acceleration: false
  prediction_interval: 300  # 5 minutes
  batch_size: 100

interfaces:
  vr_enabled: true
  webxr_enabled: true
  conversational_ai_enabled: true

external_apis:
  weather:
    provider: "openweathermap"
    api_key: "your_api_key"
  traffic:
    provider: "google_maps"
    api_key: "your_api_key"
  air_quality:
    provider: "iqair"
    api_key: "your_api_key"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/app/logs/regenerax.log"
  max_size: "100MB"
  backup_count: 5
```

## Database Setup

### SQLite (Default)

SQLite is used by default for simplicity. No additional setup required.

### PostgreSQL (Recommended for Production)

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE regenerax;
CREATE USER regenerax_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE regenerax TO regenerax_user;
\q

# Update environment variable
export DATABASE_URL="postgresql://regenerax_user:your_secure_password@localhost:5432/regenerax"
```

### MySQL

```bash
# Install MySQL
sudo apt-get install mysql-server

# Create database and user
mysql -u root -p
CREATE DATABASE regenerax;
CREATE USER 'regenerax_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON regenerax.* TO 'regenerax_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;

# Update environment variable
export DATABASE_URL="mysql://regenerax_user:your_secure_password@localhost:3306/regenerax"
```

## External Dependencies

### Redis (Optional, for Caching)

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

### NGINX (for Production)

```nginx
# /etc/nginx/sites-available/regenerax
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Verification

After installation, verify that RegeneraX is working correctly:

### 1. Health Check

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "sensors": "healthy",
    "ai_engine": "healthy"
  }
}
```

### 2. Dashboard Access

Open your browser and navigate to:
- Main Dashboard: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- WebSocket Test: http://localhost:8080

### 3. VR Interface Test

For VR capabilities, ensure:
1. Your browser supports WebXR
2. Navigate to the VR interface: http://localhost:8000/vr
3. Click "Enter VR" to test immersive features

### 4. Conversational AI Test

```python
import asyncio
from interfaces.conversational_ai import ConversationalInterface

async def test_ai():
    interface = ConversationalInterface()
    await interface.initialize()

    session_id = interface.create_session('test_user', 'citizen')
    response = await interface.chat(session_id, "Hello, how can I make my building more sustainable?")
    print(response)

asyncio.run(test_ai())
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   ```

2. **Database Connection Error**
   ```bash
   # Check database status
   systemctl status postgresql
   # Restart if needed
   sudo systemctl restart postgresql
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   htop
   # Reduce batch size in config
   export PREDICTION_BATCH_SIZE="50"
   ```

4. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install --upgrade -r requirements.txt
   ```

### Logs

Check application logs for detailed error information:

```bash
# View live logs
tail -f /app/logs/regenerax.log

# Docker logs
docker logs -f <container_name>

# Search for errors
grep -i error /app/logs/regenerax.log
```

## Performance Optimization

### For Large Cities

```yaml
# High-performance configuration
database:
  pool_size: 50
  max_overflow: 100

api:
  workers: 8

ml:
  batch_size: 500
  gpu_acceleration: true

sensors:
  update_interval: 60  # Reduce frequency for large datasets
```

### Memory Optimization

```bash
# Set memory limits
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

# For Docker
docker run --memory="4g" --memory-swap="6g" regenerax:latest
```

## Security Considerations

### Production Security

1. **Use HTTPS**
   ```bash
   # Generate SSL certificates with Let's Encrypt
   certbot --nginx -d your-domain.com
   ```

2. **Database Security**
   ```bash
   # Use strong passwords
   # Enable SSL connections
   # Restrict network access
   ```

3. **API Security**
   ```yaml
   # Enable rate limiting
   api:
     rate_limit:
       requests_per_minute: 1000
       burst_size: 100
   ```

4. **Environment Variables**
   ```bash
   # Never commit sensitive data
   # Use secrets management
   # Rotate API keys regularly
   ```

## Next Steps

After successful installation:

1. Read the [API Reference](api-reference.md) to understand available endpoints
2. Check out [Use Cases](use-cases.md) for practical examples
3. Explore the [VR Interface Guide](vr-guide.md) for immersive features
4. Review [Architecture](architecture.md) for system understanding

For support, visit our [GitHub Issues](https://github.com/your-org/regenerax/issues) or contact [support@regenerax.org](mailto:support@regenerax.org).