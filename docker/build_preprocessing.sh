#!/bin/bash
# Build preprocessing Docker container

DOCKER_USERNAME="go2432"
IMAGE_NAME="spineps-preprocessing"
TAG="latest"

echo "Building preprocessing container..."
docker build -f Dockerfile.preprocessing -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

echo ""
echo "Build complete!"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run:"
echo "  docker run -it ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
