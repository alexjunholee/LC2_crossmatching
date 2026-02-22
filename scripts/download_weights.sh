#!/bin/bash
# Download pretrained LC2 weights from GitHub Releases
#
# Usage:
#   bash scripts/download_weights.sh
#   bash scripts/download_weights.sh --tag v2.0.0
#   bash scripts/download_weights.sh --output-dir weights/

set -e

REPO="alexjunholee/LC2_crossmatching"
TAG="v2.0.0"
WEIGHTS_DIR="pretrained"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --output-dir)
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/download_weights.sh [--tag TAG] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

mkdir -p "$WEIGHTS_DIR"

echo "Downloading LC2 pretrained weights from $REPO (tag: $TAG)..."

if command -v gh &> /dev/null; then
    echo "Using GitHub CLI..."
    gh release download "$TAG" -R "$REPO" -p "lc2_pretrained.pth.tar" -D "$WEIGHTS_DIR" --clobber
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L "https://github.com/$REPO/releases/download/$TAG/lc2_pretrained.pth.tar" \
         -o "$WEIGHTS_DIR/lc2_pretrained.pth.tar"
elif command -v wget &> /dev/null; then
    echo "Using wget..."
    wget "https://github.com/$REPO/releases/download/$TAG/lc2_pretrained.pth.tar" \
         -O "$WEIGHTS_DIR/lc2_pretrained.pth.tar"
else
    echo "Error: No download tool found. Install gh, curl, or wget."
    exit 1
fi

echo ""
echo "Done! Weights saved to: $WEIGHTS_DIR/lc2_pretrained.pth.tar"
echo ""
echo "Usage:"
echo "  python evaluate.py --config configs/eval_vivid.yaml --checkpoint $WEIGHTS_DIR/lc2_pretrained.pth.tar"
