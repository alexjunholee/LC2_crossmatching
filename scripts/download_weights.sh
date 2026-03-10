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

WEIGHTS=(
    "lc2_kitti360_multi.pth.tar"
    "lc2_kitti360.pth.tar"
    "lc2_vivid.pth.tar"
    "lc2_helipr.pth.tar"
)

echo "Downloading LC2 pretrained weights from $REPO (tag: $TAG)..."

for w in "${WEIGHTS[@]}"; do
    if [[ -f "$WEIGHTS_DIR/$w" ]]; then
        echo "  Skipping $w (already exists)"
        continue
    fi
    echo "  Downloading $w..."
    if command -v gh &> /dev/null; then
        gh release download "$TAG" -R "$REPO" -p "$w" -D "$WEIGHTS_DIR" --clobber
    elif command -v curl &> /dev/null; then
        curl -L "https://github.com/$REPO/releases/download/$TAG/$w" -o "$WEIGHTS_DIR/$w"
    elif command -v wget &> /dev/null; then
        wget "https://github.com/$REPO/releases/download/$TAG/$w" -O "$WEIGHTS_DIR/$w"
    else
        echo "Error: No download tool found. Install gh, curl, or wget."
        exit 1
    fi
done

echo ""
echo "Done! Weights saved to: $WEIGHTS_DIR/"
ls -lh "$WEIGHTS_DIR/"*.pth.tar 2>/dev/null
echo ""
echo "Recommended (multi-seq, R@1=91%):"
echo "  python eval_bidirectional.py --config configs/train_kitti360_multi.yaml \\"
echo "      --checkpoint $WEIGHTS_DIR/lc2_kitti360_multi.pth.tar --gem --sequences 0000 0009"
