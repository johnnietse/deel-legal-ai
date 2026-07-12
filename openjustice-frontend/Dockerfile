# OpenJustice.ai Frontend — Multi-stage Dockerfile
# Stage 1: Build the React + Vite application
# Stage 2: Serve with nginx

# ===========================
# Builder Stage
# ===========================
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files for layer caching
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source code
COPY tsconfig.json vite.config.ts* index.html ./
COPY public/ ./public/
COPY src/ ./src/

# Build the application
ARG VITE_API_BASE=https://openjustice-api-proxy.johnnietse39.workers.dev/api
ARG VITE_AUTH_DOMAIN=openjustice.ai
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_AUTH_DOMAIN=$VITE_AUTH_DOMAIN

RUN npm run build

# ===========================
# Production Stage
# ===========================
FROM nginx:1.27-alpine AS production

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Security: drop root, run as non-root nginx user
RUN chown -R nginx:nginx /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html \
    && chown -R nginx:nginx /var/cache/nginx \
    && chown -R nginx:nginx /var/log/nginx \
    && touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/run/nginx.pid

USER nginx

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/nginx-health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
