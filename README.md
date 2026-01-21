# Pregnancy App Deployment Guide

This project is a MERN stack application with machine learning services for maternal health monitoring.

## Architecture

- **Backend Server**: Node.js Express server (port 5000)
- **ML Service**: Python Flask service (port 5001)
- **Frontend**: Static HTML/CSS/JS files served from public/

## Containerization

The project has been containerized with Dockerfiles:

- `server/Dockerfile`: For the Node.js backend
- `server/ml_service/Dockerfile`: For the Python ML service

## Deployment Options

### 1. Local Development
Run the server locally:
```bash
npm run dev
```

### 2. Docker Deployment
Build and run with Docker:

```bash
# Build images
docker build -t pregnancy-server ./server
docker build -t pregnancy-ml ./server/ml_service

# Run containers
docker run -p 5000:5000 pregnancy-server
docker run -p 5001:5001 pregnancy-ml
```

### 3. Cloud Deployment

#### Heroku (for Node.js backend)
1. Create a Heroku app
2. Set environment variables: MONGO_URI, JWT_SECRET
3. Deploy the `server/` directory
4. For ML service, deploy separately or integrate

#### Vercel (for full stack)
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root
3. Configure for Node.js

#### Azure Container Apps
1. Install Azure CLI and login
2. Create resource group and container registry
3. Build and push Docker images
4. Create container apps for each service

#### Other options: Railway, Render, Google Cloud Run

## Environment Variables

- MONGO_URI: MongoDB connection string
- JWT_SECRET: Secret for JWT tokens
- PORT: Server port (defaults to 5000)

## Database

The app uses MongoDB. For production, use MongoDB Atlas or another cloud MongoDB service.