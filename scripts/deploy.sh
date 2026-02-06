#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

set -a && source .env && set +a

# API certificate domain (defaults to api.$DOMAIN)
API_DOMAIN=${API_DOMAIN:-api.$DOMAIN}

# 1. Upload files via rsync.
rsync -av -e "ssh -i '$private_key_file'" \
  --checksum \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ruff_cache/' \
  --exclude='.venv/' \
  --exclude='.git/' \
  --exclude='.gitignore' \
  --exclude='.pre-commit-config.yaml' \
  --exclude='todo.md' \
  --exclude='*.log' \
  --exclude='.cache/' \
  --exclude='tests/' \
  --exclude='data/datasets/' \
  "$local_project_path/" "$user@$host:$remote_project_path/"

# 2. Build the docker image and start it as a daemon.
ssh -i "$private_key_file" "$user@$host" << REMOTE_COMMANDS
  echo "=== Checking disk space before cleanup ==="
  df -h

  cd "$remote_project_path"

  echo "=== Cleaning up Docker resources ==="
  # Stop all containers first
  docker stop \$(docker ps -aq) 2>/dev/null || true

  # Remove all containers, images, networks, and volumes
  docker system prune -af --volumes

  # Remove dangling images and build cache
  docker builder prune -af

  # Clean up caches
  rm -rf ~/.cache/uv

  echo "=== Disk space after cleanup ==="
  df -h

  docker compose down

  # Process nginx config AFTER cleanup to ensure it's not cleared
  echo "=== Generating nginx configuration ==="
  sed "s/\\\${DOMAIN}/$API_DOMAIN/g" ./nginx.conf > /tmp/nginx.conf

  # Verify nginx config was created successfully
  if [ ! -f /tmp/nginx.conf ] || [ ! -s /tmp/nginx.conf ]; then
    echo "ERROR: Failed to create /tmp/nginx.conf or file is empty"
    ls -lh /tmp/nginx.conf || echo "File does not exist"
    exit 1
  fi

  echo "Nginx config generated successfully at /tmp/nginx.conf"
  echo "First few lines of nginx config:"
  head -5 /tmp/nginx.conf

  # Get SSL certificate if it doesn't exist (using getssl + HTTP-01 webroot)
  if [ ! -d "./certbot/conf/live/$API_DOMAIN" ]; then
    echo "SSL certificate not found. Obtaining certificate with getssl..."

    # Ensure webroot exists for HTTP-01 challenges
    mkdir -p "$remote_project_path/certbot/www"

    # Install getssl if not present
    if ! command -v getssl >/dev/null 2>&1; then
      echo "Installing getssl..."
      curl -s https://raw.githubusercontent.com/srvrco/getssl/master/getssl -o /usr/local/bin/getssl
      chmod 700 /usr/local/bin/getssl
    fi

    # Initialize config (creates ~/.getssl/$API_DOMAIN/getssl.cfg)
    getssl -c "$API_DOMAIN" || true

    # Write a minimal config for HTTP-01 and SANs
    cat > "$HOME/.getssl/$API_DOMAIN/getssl.cfg" << EOF
CA="https://acme-v02.api.letsencrypt.org/directory"
ACCOUNT_KEY_LENGTH=4096
PRIVATE_KEY_ALG="rsa"
RENEW_ALLOW="30"
  SANS=""
  ACL=("$remote_project_path/certbot/www")
USE_SINGLE_ACL="true"
  DOMAIN_KEY_LOCATION="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.key"
  DOMAIN_CERT_LOCATION="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.crt"
  DOMAIN_CHAIN_LOCATION="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.chain.crt"
  DOMAIN_PEM_LOCATION="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.fullchain.crt"
EOF

    # Request/renew certificate
    getssl "$API_DOMAIN"

    # Install certificate to expected location
    mkdir -p ./certbot/conf/live/$API_DOMAIN

    FULLCHAIN_SRC="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.fullchain.crt"
    [ -f "$FULLCHAIN_SRC" ] || FULLCHAIN_SRC="$HOME/.getssl/$API_DOMAIN/fullchain.crt"

    CHAIN_SRC="$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.chain.crt"
    [ -f "$CHAIN_SRC" ] || CHAIN_SRC="$HOME/.getssl/$API_DOMAIN/chain.crt"

    cp "$FULLCHAIN_SRC" ./certbot/conf/live/$API_DOMAIN/fullchain.pem
    cp "$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.key" ./certbot/conf/live/$API_DOMAIN/privkey.pem
    cp "$HOME/.getssl/$API_DOMAIN/$API_DOMAIN.crt" ./certbot/conf/live/$API_DOMAIN/cert.pem
    cp "$CHAIN_SRC" ./certbot/conf/live/$API_DOMAIN/chain.pem

    echo "Certificate obtained successfully!"
  fi

  # Start all services with SSL-enabled nginx
  echo "=== Building and starting containers ==="
  docker compose up --build -d app

  # Wait for app to be ready
  echo "Waiting for app container to be ready..."
  sleep 5

  # Start nginx
  docker compose up -d nginx

  # Wait a moment for nginx to start
  sleep 3

  # Check if nginx started successfully
  if ! docker compose ps nginx | grep -q "Up"; then
    echo "ERROR: Nginx failed to start. Checking logs..."
    docker compose logs nginx
    echo "Attempting to restart nginx..."
    docker compose restart nginx
    sleep 3
  fi

  # Final status check
  echo "=== Container Status ==="
  docker compose ps

  # Verify nginx is responding
  echo "=== Testing nginx connectivity ==="
  if docker exec ai-detector-nginx nginx -t 2>&1; then
    echo "✓ Nginx configuration is valid"
  else
    echo "✗ Nginx configuration has errors"
  fi

  # Test internal connectivity
  if docker exec ai-detector-app curl -s -o /dev/null -w "%{http_code}" http://localhost:7123 | grep -q "200\|404\|405"; then
    echo "✓ App is responding on port 7123"
  else
    echo "✗ App is not responding"
  fi

  echo "=== Deployment complete. Tailing logs (Ctrl+C to exit) ==="
  docker compose logs --follow app nginx
REMOTE_COMMANDS
