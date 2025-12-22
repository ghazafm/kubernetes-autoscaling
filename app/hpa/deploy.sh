#!/usr/bin/env bash
set -euo pipefail

# Make executable by default permissions when added to repo - if your git umask
# prevents executable bit, run: chmod +x app/hpa/deploy-latest.sh

# Deploy the newest image to the cluster by updating the Deployment's container image
# and triggering a rollout. Useful for CI or quick deploys when image is already pushed.

IMAGE_NAME_DEFAULT="fauzanghaza/flask-app"
IMAGE_TAG_DEFAULT="latest"
DEPLOYMENT_DEFAULT="hpa-flask-app"
CONTAINER_DEFAULT="hpa-flask-app"
NAMESPACE_DEFAULT="default"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --image NAME        Image name (default: ${IMAGE_NAME_DEFAULT})
  --tag TAG           Image tag (default: ${IMAGE_TAG_DEFAULT})
  --deployment NAME   Deployment name (default: ${DEPLOYMENT_DEFAULT})
  --container NAME    Container name in the deployment (default: ${CONTAINER_DEFAULT})
  --namespace NAME    Kubernetes namespace (default: ${NAMESPACE_DEFAULT})
  --wait              Wait for rollout to finish (default: true)
  -h, --help          Show this help

Example:
  $0 --image myrepo/flask-app --tag v1.2.3 --deployment test-flask-app
EOF
}

IMAGE="${IMAGE_NAME_DEFAULT}"
TAG="${IMAGE_TAG_DEFAULT}"
DEPLOYMENT="${DEPLOYMENT_DEFAULT}"
CONTAINER="${CONTAINER_DEFAULT}"
NAMESPACE="${NAMESPACE_DEFAULT}"
WAIT=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --deployment) DEPLOYMENT="$2"; shift 2;;
    --container) CONTAINER="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --wait) WAIT=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

FULL_IMAGE="$IMAGE:$TAG"

echo "Deploying image: $FULL_IMAGE to deployment/$DEPLOYMENT (container: $CONTAINER) in namespace $NAMESPACE"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found in PATH" >&2
  exit 1
fi

# Update the image on the deployment's container
echo "Updating deployment image..."
kubectl set image deployment/${DEPLOYMENT} ${CONTAINER}=${FULL_IMAGE} -n ${NAMESPACE} --record || true

# Patch deployment template annotation with a timestamp to force pod restart
TS=$(date -u +%s)
echo "Patching deployment to force rollout (annotation deployTimestamp=${TS})..."
kubectl patch deployment/${DEPLOYMENT} -n ${NAMESPACE} -p \
  "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"deployTimestamp\":\"${TS}\"}}}}}"

if [ "$WAIT" = true ]; then
  echo "Waiting for rollout to complete..."
  kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE}
fi

echo "Deployment updated."
