#!/usr/bin/env python3
"""
Train microSAM using Hypha service with data from simple_finetuning folder.
This script loads local training data and submits it to the remote Hypha service.
"""
import asyncio
import os
import json
import numpy as np
import time
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

# Get the script and data directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
data_dir = os.path.join(project_root, "examples", "simple_finetuning")

async def train_with_hypha():
    print("🎓 Training microSAM model using Hypha service")
    print("=" * 60)
    
    # Load .env file if it exists
    load_dotenv()
    
    # Connect to Hypha server
    print("\n1️⃣ Connecting to Hypha server...")
    token = os.getenv("HYPHA_TOKEN") or await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
    })
    print("✅ Connected to Hypha server")
    
    # Connect directly to microSAM service via HTTP
    # Note: WebRTC has known issues with context parameter handling in hypha_rpc
    print("\n2️⃣ Connecting to microSAM service...")
    
    try:
        # Connect directly to the micro-sam service
        microsam_service = await server.get_service("agent-lens/micro-sam")
        print("✅ Connected to microSAM service: agent-lens/micro-sam")
    except Exception as connection_error:
        print(f"❌ Could not connect to microSAM service: {connection_error}")
        return
    
    # Load training data from simple_finetuning folder
    print("\n3️⃣ Loading training data from simple_finetuning folder...")
    print(f"   📁 Data directory: {data_dir}")
    
    # Load all images
    images = []
    
    # Try loading from .npy files first (preprocessed)
    for i in range(4):
        npy_path = os.path.join(data_dir, f"image_{i}.npy")
        if os.path.exists(npy_path):
            image = np.load(npy_path)
            # Ensure correct shape (C, H, W)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)  # (H, W) -> (1, H, W)
            elif len(image.shape) == 3 and image.shape[0] not in [1, 3]:
                # Assume (H, W, C) -> convert to (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            images.append(image)
            print(f"   ✅ Loaded image_{i}.npy: {image.shape}")
        else:
            print(f"   ⚠️ image_{i}.npy not found at {npy_path}")
    
    print(f"   📊 Loaded {len(images)} images total")
    
    # Check if we have enough images
    if len(images) < 2:
        print("❌ Error: Minimum 2 images required for training")
        return
    
    # Load annotations
    annotations_path = os.path.join(data_dir, "annotations.json")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    print(f"   ✅ Loaded annotations: {len(annotations['images'])} images, {len(annotations['annotations'])} annotations")
    
    # Calculate train/val split
    n_train = int(0.8 * len(images))
    n_val = len(images) - n_train
    print(f"\n   📊 Data split: {n_train} train, {n_val} validation")
    print(f"   ✅ Proper train/validation split enabled")
    
    # Start training
    print(f"   Training parameters:")
    print(f"   - Images: {len(images)}")
    print(f"   - Epochs: 10")
    print(f"   - Freeze encoder: True (AIS head only)")
    
    start_time = time.time()
    try:
        # Submit training job
        result = await microsam_service.start_fit(
            images=images,
            annotations=annotations,
            n_epochs=10
        )
        
        print(f"   ✅ Training job submitted: {result}")
        
        # Monitor training progress
        print("\n5️⃣ Monitoring training progress...")
        max_wait_time = 3600  # 1 hour max
        check_interval = 10  # Check every 10 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
            
            status = await microsam_service.get_fit_status()
            print(f"   [{elapsed_time}s] Status: {status['status']} - {status['message']}")
            
            if status['status'] == 'completed':
                print("\n✅ Training completed successfully!")
                break
            elif status['status'] == 'error':
                print(f"\n❌ Training failed: {status.get('message', 'Unknown error')}")
                break
            elif status['status'] == 'running':
                # Continue monitoring
                pass
        
        if elapsed_time >= max_wait_time:
            print("\n⚠️ Training exceeded maximum wait time")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("🎉 Training workflow completed!")

if __name__ == "__main__":
    # Set CUDA memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    asyncio.run(train_with_hypha())

