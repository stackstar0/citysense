# RegeneraX - Full Stack Architecture Setup

## 🌱 RegeneraX Complete Platform

This document outlines the complete full-stack architecture for RegeneraX, integrating both the Python backend (citysense) and React frontend (living-city-mind).

## 📁 Project Structure

```
RegeneraX-Platform/
├── backend/                    # Python FastAPI Backend
│   ├── api/                   # API endpoints
│   ├── core/                  # Core urban intelligence
│   ├── ai_engine/            # AI/ML components (dependency-free)
│   ├── interfaces/           # Chat, VR interfaces
│   ├── ecosystem/            # Impact analysis
│   ├── regenerative/         # Optimization
│   └── visualization/        # Web dashboards
├── frontend/                  # React TypeScript Frontend
│   ├── src/components/       # React components
│   ├── src/pages/           # Page components
│   ├── src/hooks/           # Custom hooks
│   └── src/integrations/    # Backend API integration
└── shared/                   # Shared types and utilities
    ├── types/               # TypeScript definitions
    └── api/                 # API schemas
```

## 🔧 Technology Stack

### Backend (Python)
- **Framework**: FastAPI
- **AI/ML**: Custom implementations (no sklearn dependency)
- **Database**: SQLite with async support
- **Real-time**: WebSocket for live data
- **API**: RESTful with OpenAPI docs

### Frontend (React)
- **Framework**: React 18 + TypeScript
- **UI Library**: shadcn/ui + Tailwind CSS
- **State Management**: TanStack Query
- **Charts**: Chart.js / Recharts
- **Build Tool**: Vite

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-minimal.txt
python server.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🌐 API Integration

The frontend connects to the backend via:
- REST API: `http://localhost:9000/api/`
- WebSocket: `ws://localhost:9000/ws`
- Static files: `http://localhost:9000/`

## 📊 Features

### Backend Features
- ✅ Real-time city vital signs monitoring
- ✅ AI-powered pattern recognition (dependency-free)
- ✅ Regenerative design recommendations
- ✅ Conversational AI interface
- ✅ VR/AR data integration
- ✅ Impact analysis and optimization

### Frontend Features
- ✅ Interactive dashboard with live charts
- ✅ AI consultant chat interface
- ✅ Building assessment tools
- ✅ Ecosystem flow visualization
- ✅ Responsive design with dark/light themes
- ✅ Real-time data updates

## 🔗 Integration Points

1. **API Layer**: Frontend calls backend REST endpoints
2. **WebSocket**: Real-time data streaming
3. **Shared Types**: TypeScript definitions for data models
4. **Authentication**: JWT-based (future enhancement)
5. **File Upload**: Building data and images (future enhancement)