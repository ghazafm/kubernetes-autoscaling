#!/bin/bash

# Flask App Deployment Script
# This script builds, pushes, and deploys the Flask application to Kubernetes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="fauzanghaza/flask-app"
IMAGE_TAG="latest"
APP_NAME="flask-app"
NAMESPACE="default"

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Flask Application Deployment Script                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}▶ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if kubectl is installed
print_section "Checking prerequisites..."
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl first."
    exit 1
fi
print_success "kubectl found"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "docker not found. Please install Docker first."
    exit 1
fi
print_success "docker found"

# Check Kubernetes connection
print_section "Checking Kubernetes connection..."
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi
print_success "Connected to Kubernetes cluster"

# Detect cluster architectures
print_section "Detecting cluster node architectures..."
ARCHS=$(kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}' | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
echo -e "   Detected architectures: ${GREEN}$ARCHS${NC}"

# Build multi-platform Docker image and push (creates a multi-arch manifest)
print_section "Building multi-platform Docker image for: ${ARCHS:-all}..."
echo "   Image: $IMAGE_NAME:$IMAGE_TAG"

# Ensure buildx is available
if ! docker buildx version &> /dev/null; then
    print_error "docker buildx is not available. Please enable BuildKit or install Docker Buildx."
    exit 1
fi

# Create and use a builder (idempotent)
docker buildx create --use --name buildx_builder >/dev/null 2>&1 || true

# Build for common architectures (amd64 and arm64). Use the detected ARCHS to decide if necessary.
PLATFORMS="linux/amd64,linux/arm64"
print_section "Running buildx build --platform $PLATFORMS --push ..."
if docker buildx build --platform $PLATFORMS -t $IMAGE_NAME:$IMAGE_TAG . --push; then
    print_success "Docker image built and pushed successfully (multi-arch)"
else
    print_error "Failed to build/push multi-platform Docker image"
    echo "   Make sure you're logged in: docker login and that buildx is configured correctly"
    exit 1
fi

# Optional: Deploy Prometheus ServiceMonitor (only if Prometheus Operator is installed)
print_section "Checking for Prometheus Operator..."
if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
    print_success "Prometheus Operator detected"
    if [ -f monitor.yaml ]; then
        echo "   Deploying ServiceMonitor..."
        kubectl apply -f monitor.yaml
        print_success "ServiceMonitor deployed"
    else
        echo -e "${YELLOW}   monitor.yaml not found, skipping${NC}"
    fi
else
    echo -e "${YELLOW}   Prometheus Operator not installed, skipping ServiceMonitor deployment${NC}"
fi

# Deploy ConfigMap first
print_section "Deploying ConfigMap..."
if [ -f configmap.yaml ]; then
    if kubectl apply -f configmap.yaml; then
        print_success "ConfigMap deployed"
    else
        print_error "Failed to deploy ConfigMap"
        exit 1
    fi
else
    echo -e "${YELLOW}   configmap.yaml not found, using inline environment variables${NC}"
fi

# Deploy application (use deployment.yaml if exists, fallback to app.yaml)
print_section "Deploying application to Kubernetes..."
if [ -f deployment.yaml ]; then
    DEPLOY_FILE="deployment.yaml"
    echo "   Using $DEPLOY_FILE (with ConfigMap support)"
else
    DEPLOY_FILE="app.yaml"
    echo "   Using $DEPLOY_FILE (with inline environment variables)"
fi

if kubectl apply -f $DEPLOY_FILE; then
    print_success "Application deployed"
else
    print_error "Failed to deploy application"
    exit 1
fi

# Force rollout restart to pull new multi-arch image (in case old pods exist with wrong arch)
print_section "Restarting deployment to use new image..."
kubectl rollout restart deployment/$APP_NAME
echo "   Waiting for rollout to complete..."
if kubectl rollout status deployment/$APP_NAME --timeout=120s; then
    print_success "Rollout completed successfully"
else
    print_error "Rollout did not complete in time"
fi

# Wait for deployment to be ready
print_section "Waiting for pods to be ready..."
echo "   This may take a minute..."
if kubectl wait --for=condition=ready pod -l app=$APP_NAME --timeout=120s; then
    print_success "Pods are ready"
else
    print_error "Pods failed to become ready"
    echo -e "\n${YELLOW}Checking pod status:${NC}"
    kubectl get pods -l app=$APP_NAME
    echo -e "\n${YELLOW}Pod logs:${NC}"
    kubectl logs -l app=$APP_NAME --tail=50
    exit 1
fi

# Wait for metrics to be available
print_section "Waiting for metrics to be available..."
echo "   This may take 1-2 minutes..."
RETRY_COUNT=0
MAX_RETRIES=12
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if kubectl top pods -l app=$APP_NAME &> /dev/null; then
        print_success "Metrics are available"
        kubectl top pods -l app=$APP_NAME
        break
    fi
    echo -n "."
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "Metrics not available after waiting"
    echo "   You may need to wait longer before deploying HPA"
fi

# Display deployment status
print_section "Deployment Status"
echo ""
echo -e "${GREEN}Pods:${NC}"
kubectl get pods -l app=$APP_NAME

echo ""
echo -e "${GREEN}Service:${NC}"
kubectl get svc $APP_NAME

echo ""
echo -e "${GREEN}Resource Usage:${NC}"
kubectl top pods -l app=$APP_NAME 2>/dev/null || echo "   Metrics not yet available"

# Get service endpoint
print_section "Access Information"
SERVICE_TYPE=$(kubectl get svc $APP_NAME -o jsonpath='{.spec.type}')

if [ "$SERVICE_TYPE" = "LoadBalancer" ]; then
    EXTERNAL_IP=$(kubectl get svc $APP_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$EXTERNAL_IP" ]; then
        EXTERNAL_IP="<pending>"
        echo -e "${YELLOW}⚠  LoadBalancer IP is pending. Check later with: kubectl get svc $APP_NAME${NC}"
    else
        echo -e "${GREEN}External URL: http://$EXTERNAL_IP:5000${NC}"
        echo -e "   API: http://$EXTERNAL_IP:5000/api"
        echo -e "   CPU Test: http://$EXTERNAL_IP:5000/api/cpu?iterations=1000000"
        echo -e "   Memory Test: http://$EXTERNAL_IP:5000/api/memory?size_mb=100"
        echo -e "   Metrics: http://$EXTERNAL_IP:8000/metrics"
    fi
else
    echo -e "${YELLOW}Service type is $SERVICE_TYPE${NC}"
fi

echo ""
echo -e "${BLUE}Port Forward:${NC}"
echo "   kubectl port-forward svc/$APP_NAME 5000:5000"
echo "   Then access: http://localhost:5000/api"

# Display next steps
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Deployment Complete!                                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Test the application:"
echo "   kubectl port-forward svc/$APP_NAME 5000:5000"
echo "   curl http://localhost:5000/api"
echo "   curl http://localhost:5000/health"
echo ""
echo "2. Check metrics:"
echo "   kubectl port-forward svc/$APP_NAME 8000:8000"
echo "   curl http://localhost:8000/metrics"
echo ""
echo "3. Update configuration:"
echo "   kubectl edit configmap flask-app-config"
echo "   kubectl rollout restart deployment/$APP_NAME"
echo ""
echo "4. Deploy HPA (after metrics are available):"
echo "   kubectl apply -f hpa.yaml"
echo "   kubectl get hpa -w"
echo ""
echo "5. Run load tests:"
echo "   cd ../test"
echo "   ./run-k6.sh"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "   kubectl get pods -l app=$APP_NAME -w    # Watch pods"
echo "   kubectl logs -l app=$APP_NAME -f        # Follow logs"
echo "   kubectl top pods -l app=$APP_NAME       # Resource usage"
echo "   kubectl get configmap flask-app-config  # View config"
echo "   kubectl delete -f deployment.yaml       # Delete deployment"
echo ""
