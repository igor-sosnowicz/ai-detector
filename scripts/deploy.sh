#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

# My favorite from the comments. Thanks @richarddewit & others!
set -a && source .env && set +a

# 1. Upload files via rsync.
rsync -av -e "ssh -i '$private_key_file'" \
  --checksum \
  --exclude='src/__pycache__' \
  --exclude='.ruff_cache' \
  --exclude='todo.md' \
  --exclude='.git' \
  --exclude='scripts' \
  --exclude='src/tests' \
  --exclude='.pre-commit-config.yaml' \
  --exclude='.venv' \
  "$local_project_path/" "$user@$host:$remote_project_path/"

# 2. Build the docker image and start it as a daemon.
ssh -i "$private_key_file" "$user@$host" << REMOTE_COMMANDS
  echo "=== Checking disk space before cleanup ==="
  df -h

  echo "=== Cleaning up Docker resources ==="
  # Stop all containers first
  docker stop \$(docker ps -aq) 2>/dev/null || true

  # Remove all containers, images, networks, and volumes
  docker system prune -af --volumes

  # Remove dangling images and build cache
  docker builder prune -af

  # Clean up other caches
  rm -rf ~/.cache/uv
  rm -rf /tmp/*

  echo "=== Disk space after cleanup ==="
  df -h

  cd "$remote_project_path"
  docker compose down

  # Get SSL certificate if it doesn't exist (using lightweight acme.sh)
  if [ ! -d "./certbot/conf/live/$DOMAIN" ]; then
    echo "SSL certificate not found. Obtaining certificate with acme.sh..."

    # Install acme.sh if not present (lightweight, pure shell script)
    if [ ! -d ~/.acme.sh ]; then
      echo "Installing acme.sh..."
      curl https://get.acme.sh | sh -s email=igor@sosnowicz.eu
    fi

    # Get certificate using standalone mode (port 80)
    ~/.acme.sh/acme.sh --issue -d $DOMAIN --standalone --httpport 80

    # Install certificate to expected location
    mkdir -p ./certbot/conf/live/$DOMAIN
    ~/.acme.sh/acme.sh --install-cert -d $DOMAIN \
      --fullchain-file ./certbot/conf/live/$DOMAIN/fullchain.pem \
      --key-file ./certbot/conf/live/$DOMAIN/privkey.pem \
      --cert-file ./certbot/conf/live/$DOMAIN/cert.pem \
      --ca-file ./certbot/conf/live/$DOMAIN/chain.pem

    echo "Certificate obtained successfully!"
  fi

  # Start all services with SSL-enabled nginx
  docker compose up --build -d nginx app

  docker compose logs --follow app nginx
REMOTE_COMMANDS
