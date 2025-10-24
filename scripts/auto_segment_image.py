#!/usr/bin/env python3
"""
Auto-segmentation of microscopy images using deployed microSAM service.
"""
import asyncio
import os
import numpy as np
import imageio.v3 as imageio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

async def auto_segment_image():
    print("🔬 Auto-segmentation of microscopy images using microSAM")
    
    # Load .env file if it exists
    load_dotenv()
    
    # Connect to Hypha server
    print("Connecting to Hypha server...")
    token = os.getenv("HYPHA_TOKEN") or await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
    })
    
    # Load the test image
    print("Loading test image...")
    image_path = "examples/simple_segmentation/test_image.png"
    image = imageio.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    # Convert image to the format expected by microSAM (C, H, W)
    if len(image.shape) == 2:
        # Grayscale image: (H, W) -> (1, H, W)
        image_array = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        # RGB image: (H, W, C) -> (C, H, W)
        image_array = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    print(f"Converted image shape: {image_array.shape}")
    
    # Connect directly to the micro-sam application
    print("Connecting to microSAM service...")
    try:
        # Try to connect directly to the micro-sam application
        microsam_service = await server.get_service("agent-lens/micro-sam")
        print("✅ Connected to microSAM service: agent-lens/micro-sam")
    except Exception as e:
        print(f"❌ Could not connect to microSAM service: {e}")
        return
    
    # Perform auto-segmentation
    print("🎯 Performing auto-segmentation...")
    try:
        # Use the segment_all method for zero-prompt auto-segmentation
        segmentation_result = await microsam_service.segment_all(
            image_or_embedding=image_array,
            embedding=False
        )
        
        print("✅ Auto-segmentation completed!")
        print(f"Segmentation result type: {type(segmentation_result)}")
        
        if hasattr(segmentation_result, 'shape'):
            print(f"Segmentation shape: {segmentation_result.shape}")
        elif isinstance(segmentation_result, dict):
            print(f"Segmentation keys: {list(segmentation_result.keys())}")
        else:
            print(f"Segmentation result: {segmentation_result}")
            
        # Save the segmentation result
        output_path = "examples/simple_segmentation/segmented_result.png"
        if hasattr(segmentation_result, 'shape') and len(segmentation_result.shape) >= 2:
            # Convert to uint8 for saving
            if segmentation_result.dtype != np.uint8:
                # Normalize to 0-255 range
                seg_normalized = ((segmentation_result - segmentation_result.min()) / 
                                (segmentation_result.max() - segmentation_result.min()) * 255).astype(np.uint8)
            else:
                seg_normalized = segmentation_result
            
            imageio.imwrite(output_path, seg_normalized)
            print(f"💾 Segmentation saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return
    
    print("🎉 Auto-segmentation completed successfully!")

if __name__ == "__main__":
    asyncio.run(auto_segment_image())
