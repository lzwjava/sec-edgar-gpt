#!/bin/bash
# Deploy to Cloudflare Workers
# Usage: CLOUDFLARE_API_TOKEN=your_token ./deploy.sh
set -e

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
  echo "Error: CLOUDFLARE_API_TOKEN not set"
  echo "Usage: CLOUDFLARE_API_TOKEN=your_token ./deploy.sh"
  exit 1
fi

cd "$(dirname "$0")/.."
NODE_OPTIONS="--dns-result-order=ipv4first" wrangler deploy
