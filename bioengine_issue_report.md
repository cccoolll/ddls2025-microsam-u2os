# BioEngine Deployment Issue Report

## Date
2026-03-23

## Issue Summary
**Cannot initialize BioEngine worker due to connection limit exceeded.**

## Error Details
```
Failed to start BioEngine worker: Connection limit exceeded: 
user rapid-polo-74876236 has 200/200 connections
```

## Root Cause
The Hypha server (hypha.aicell.io) enforces a **200 concurrent WebSocket connection limit per user**. User `rapid-polo-74876236` has reached this limit, preventing any new connections.

## Affected Operations
1. ❌ BioEngine worker initialization
2. ❌ Service deployment via `deploy_cellsegmenter.py`
3. ❌ Any Hypha RPC connections to workspace `reef-imaging`

## Steps Already Completed
- ✅ Updated bioengine: `0.6.5` → `0.6.7`
- ✅ Updated HYPHA_TOKEN for `reef-imaging` workspace
- ❌ BioEngine worker start: **FAILED** (connection limit)
- ⏸️ Deployment pending: Waiting for worker

## Proposed Solutions

### Option 1: Wait (Recommended First)
Stale connections may auto-expire. Retry deployment in 10-30 minutes.

### Option 2: Admin Intervention (Fastest)
Contact Hypha server admin to manually clear stale connections for:
- **User:** `rapid-polo-74876236`
- **Workspace:** `reef-imaging`
- **Server:** `https://hypha.aicell.io`

### Option 3: Alternative Token
Use a different Hypha token with fewer active connections.

## Post-Resolution Steps
Once connections are cleared:

```bash
# 1. Start BioEngine worker (Window 1)
conda activate microsam
export HYPHA_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m bioengine.worker --mode single-machine --head-num-gpus 2 --head-num-cpus 6 \
    --workspace reef-imaging --client-id cell_segmenter

# 2. Deploy CellSegmenter (Window 2)
conda activate microsam
python scripts/deploy_cellsegmenter.py
```

## Contact
- **Project:** DDLS 2025 - microSAM U2OS Segmentation
- **Workspace:** reef-imaging
- **Service ID:** cell_segmenter
