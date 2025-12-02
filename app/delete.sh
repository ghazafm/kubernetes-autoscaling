#!/bin/bash

# Flask App Deletion Script
# This script removes the Flask application and related resources from Kubernetes

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
echo -e "${BLUE}║  Flask Application Deletion Script                  ║${NC}"
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

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if kubectl is installed
print_section "Checking prerequisites..."
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl first."
    exit 1
fi
print_success "kubectl found"

# Check Kubernetes connection
print_section "Checking Kubernetes connection..."
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi
print_success "Connected to Kubernetes cluster"

# Confirmation prompt
print_section "Confirmation"
echo -e "${RED}WARNING: This will delete the following resources:${NC}"
echo "   - Deployment: $APP_NAME"
echo "   - Service: $APP_NAME"
echo "   - ConfigMap: flask-app-config"
echo "   - HPA: $APP_NAME (if exists)"
echo "   - ServiceMonitor: $APP_NAME (if exists)"
echo ""

read -p "Are you sure you want to delete these resources? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deletion cancelled.${NC}"
    exit 0
fi

# Delete HPA if exists
print_section "Deleting HPA..."
if kubectl get hpa $APP_NAME &> /dev/null; then
    if kubectl delete hpa $APP_NAME; then
        print_success "HPA deleted"
    else
        print_warning "Failed to delete HPA"
    fi
else
    echo "   HPA not found, skipping"
fi

# Delete ServiceMonitor if exists
print_section "Deleting ServiceMonitor..."
if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
    if kubectl get servicemonitor $APP_NAME &> /dev/null; then
        if kubectl delete -f monitor.yaml 2>/dev/null || kubectl delete servicemonitor $APP_NAME; then
            print_success "ServiceMonitor deleted"
        else
            print_warning "Failed to delete ServiceMonitor"
        fi
    else
        echo "   ServiceMonitor not found, skipping"
    fi
else
    echo "   Prometheus Operator not installed, skipping ServiceMonitor deletion"
fi

# Delete Deployment
print_section "Deleting Deployment..."
if [ -f deployment.yaml ]; then
    DEPLOY_FILE="deployment.yaml"
else
    DEPLOY_FILE="app.yaml"
fi

if kubectl get deployment $APP_NAME &> /dev/null; then
    if kubectl delete -f $DEPLOY_FILE 2>/dev/null || kubectl delete deployment $APP_NAME; then
        print_success "Deployment deleted"
    else
        print_error "Failed to delete Deployment"
    fi
else
    echo "   Deployment not found, skipping"
fi

# Delete ConfigMap
print_section "Deleting ConfigMap..."
if kubectl get configmap flask-app-config &> /dev/null; then
    if kubectl delete -f configmap.yaml 2>/dev/null || kubectl delete configmap flask-app-config; then
        print_success "ConfigMap deleted"
    else
        print_warning "Failed to delete ConfigMap"
    fi
else
    echo "   ConfigMap not found, skipping"
fi

# Wait for pods to terminate
print_section "Waiting for pods to terminate..."
echo "   This may take a moment..."
RETRY_COUNT=0
MAX_RETRIES=12
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    POD_COUNT=$(kubectl get pods -l app=$APP_NAME --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "$POD_COUNT" -eq 0 ]; then
        print_success "All pods terminated"
        break
    fi
    echo -n "."
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_warning "Some pods may still be terminating"
    kubectl get pods -l app=$APP_NAME 2>/dev/null || true
fi

# Verify deletion
print_section "Verification"
echo ""

echo -e "${GREEN}Remaining Resources:${NC}"
echo ""

echo "Deployments:"
kubectl get deployment -l app=$APP_NAME 2>/dev/null || echo "   None found"

echo ""
echo "Services:"
kubectl get svc $APP_NAME 2>/dev/null || echo "   None found"

echo ""
echo "Pods:"
kubectl get pods -l app=$APP_NAME 2>/dev/null || echo "   None found"

echo ""
echo "ConfigMaps:"
kubectl get configmap flask-app-config 2>/dev/null || echo "   None found"

echo ""
echo "HPA:"
kubectl get hpa $APP_NAME 2>/dev/null || echo "   None found"

# Display completion message
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Deletion Complete!                                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Resources have been removed from the cluster.${NC}"
echo ""
echo -e "${YELLOW}Note:${NC}"
echo "   - The Docker image $IMAGE_NAME:$IMAGE_TAG still exists in the registry"
echo "   - To remove the image from Docker Hub, use the Docker Hub web interface"
echo "   - To remove local Docker images:"
echo "     docker rmi $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo -e "${BLUE}To redeploy:${NC}"
echo "   ./deploy.sh"
echo ""
