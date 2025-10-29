#!/usr/bin/env python3
"""
Auto-segmentation of microscopy images using deployed microSAM service.
Images are JPEG-compressed and base64-encoded for efficient transmission.
"""
import asyncio
import os
import time
import base64
import numpy as np
import imageio.v3 as imageio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login
from PIL import Image
import io

def encode_image_to_jpeg(image_array: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode numpy array image to JPEG bytes.
    
    Args:
        image_array: Image as numpy array (H, W) for grayscale or (H, W, C) for RGB
        quality: JPEG quality (1-100), default 85
        
    Returns:
        JPEG-encoded bytes
    """
    # Normalize image to 0-255 uint8 if needed
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Convert numpy array to PIL Image
    if len(image_array.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image_array, mode='L')
    elif len(image_array.shape) == 3:
        # RGB
        pil_image = Image.fromarray(image_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")
    
    # Encode to JPEG
    jpeg_buffer = io.BytesIO()
    pil_image.save(jpeg_buffer, format='JPEG', quality=quality)
    jpeg_bytes = jpeg_buffer.getvalue()
    
    return jpeg_bytes

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
    
    # Connect directly to microSAM service via HTTP
    # Note: WebRTC has known issues with context parameter handling in hypha_rpc
    print("🌐 Connecting to microSAM service...")
    
    try:
        # Connect directly to the micro-sam service
        microsam_service = await server.get_service("agent-lens/micro-sam")
        print("✅ Connected to microSAM service: agent-lens/micro-sam")
        
    except Exception as e:
        print(f"❌ Could not connect to microSAM service: {e}")
        return
    
    # Load the test image
    print("Loading test image...")
    image_path = "examples/simple_segmentation/test_image_large.png"
    image = imageio.imread(image_path)
    print(f"Original image shape: {image.shape}")
    
    # Normalize image to 0-255 uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Calculate original size
    original_size_mb = image.nbytes / (1024 * 1024)
    print(f"Original image size: {original_size_mb:.2f} MB")
    
    # Compress to JPEG for efficient transmission
    jpeg_quality = 85
    print(f"Compressing image to JPEG (quality={jpeg_quality})...")
    jpeg_bytes = encode_image_to_jpeg(image, quality=jpeg_quality)
    compressed_size_mb = len(jpeg_bytes) / (1024 * 1024)
    compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
    
    print(f"Compressed image size: {compressed_size_mb:.2f} MB ({compression_ratio:.1f}x reduction)")
    
    # Encode JPEG bytes to base64 for safe JSON transmission
    jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    
    # Perform auto-segmentation with timing
    print(f"🎯 Performing auto-segmentation with JPEG-compressed image...")
    
    start_time = time.time()
    try:
        # Use the segment_all method for zero-prompt auto-segmentation
        # NOTE: Service ONLY accepts base64-encoded JPEG strings, NO numpy arrays
        segmentation_result = await microsam_service.segment_all(
            image_or_embedding=jpeg_base64,  # Send base64-encoded JPEG
            embedding=False
        )
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Auto-segmentation completed!")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        # Decode JPEG base64 result (service returns JPEG base64 string, NOT numpy array)
        print(f"Decoding JPEG base64 segmentation result...")
        
        if not isinstance(segmentation_result, str):
            raise ValueError(
                f"Expected JPEG base64 string result, got {type(segmentation_result)}. "
                f"Service should return JPEG base64-encoded segmentation."
            )
        
        # Decode base64 to get JPEG bytes
        jpeg_bytes = base64.b64decode(segmentation_result)
        
        # Decode JPEG bytes to image
        seg_image = Image.open(io.BytesIO(jpeg_bytes))
        seg_array = np.array(seg_image)
        
        print(f"Segmentation decoded shape: {seg_array.shape}")
        
        # Save the segmentation result
        output_path = "examples/simple_segmentation/segmented_result.png"
        imageio.imwrite(output_path, seg_array)
        print(f"💾 Segmentation saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"🎉 Auto-segmentation completed successfully!")

if __name__ == "__main__":
    asyncio.run(auto_segment_image())
