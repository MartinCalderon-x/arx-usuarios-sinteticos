# ============================================
# Product Dockerfile
# ============================================
# This Dockerfile builds and deploys ONLY the product/ directory.
# The factory/ tools are for development only - they don't get deployed.
#
# Customize this file based on your product's technology.
# See examples below for common setups.
# ============================================

# ============================================
# EXAMPLE: Python Product (FastAPI, Flask, etc.)
# Uncomment this section for Python products
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Copy and install product dependencies
COPY product/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy product code
COPY product/ .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# Run (adjust based on your framework)
# FastAPI: CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
# Flask: CMD ["python", "src/main.py"]
CMD ["python", "src/main.py"]


# ============================================
# EXAMPLE: Node.js Product (React, Next.js, Express, etc.)
# Uncomment and use this instead for Node products
# ============================================
# # Build stage
# FROM node:20-alpine as builder
# WORKDIR /app
# COPY product/package*.json ./
# RUN npm ci
# COPY product/ .
# RUN npm run build
#
# # Production stage
# FROM node:20-alpine
# WORKDIR /app
# COPY --from=builder /app/dist ./dist
# COPY --from=builder /app/node_modules ./node_modules
# COPY --from=builder /app/package.json ./
# ENV PORT=8080
# EXPOSE 8080
# CMD ["npm", "start"]


# ============================================
# EXAMPLE: Static Site (React/Vite build → Nginx)
# Uncomment and use this for static frontends
# ============================================
# # Build stage
# FROM node:20-alpine as builder
# WORKDIR /app
# COPY product/package*.json ./
# RUN npm ci
# COPY product/ .
# RUN npm run build
#
# # Serve with nginx
# FROM nginx:alpine
# COPY --from=builder /app/dist /usr/share/nginx/html
# COPY product/nginx.conf /etc/nginx/conf.d/default.conf
# EXPOSE 8080
# CMD ["nginx", "-g", "daemon off;"]
