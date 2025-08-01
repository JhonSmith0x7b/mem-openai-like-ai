#!/bin/bash

set -ex
# image_name="mem-openai-like-ai:$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
image_name="mem-openai-like-ai:latest"
docker build --platform linux/amd64  -t ${image_name} .