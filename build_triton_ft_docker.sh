cd fastertransformer_backend
export WORKSPACE=$(pwd)
export CONTAINER_VERSION=22.12
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
docker build --rm   \
    --build-arg TRITON_VERSION=${CONTAINER_VERSION}   \
    -t ${TRITON_DOCKER_IMAGE} \
    -f docker/Dockerfile \
    .
