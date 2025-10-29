#!/usr/bin/env python3
"""
Test OME-Zarr segmentation using deployed microSAM service.
Segments cells from OME-Zarr data on Hypha platform and generates polygon annotations.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login


async def test_ome_zarr_segmentation():
    print("🔬 Testing OME-Zarr Segmentation with microSAM")
    
    # Load .env file if it exists
    load_dotenv()
    
    # Connect to Hypha server
    print("Connecting to Hypha server...")
    token = os.getenv("HYPHA_TOKEN") or await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
    })
    
    # Connect to microSAM service
    print("🌐 Connecting to microSAM service...")
    
    try:
        # Connect directly to the micro-sam service
        microsam_service = await server.get_service("agent-lens/micro-sam")
        print("✅ Connected to microSAM service: agent-lens/micro-sam")
        
    except Exception as e:
        print(f"❌ Could not connect to microSAM service: {e}")
        return
    
    # Test OME-Zarr URL from Hypha platform
    zarr_url = "https://hypha.aicell.io/agent-lens/artifacts/hpa-example-sample-20250114-150051/zip-files/well_C3_96.zip/~/data.zarr/"
    well_id = "C3"
    channel_idx = 1  # Fluorescence 405 nm Ex
    
    print(f"\n📊 Test Parameters:")
    print(f"  Zarr URL: {zarr_url}")
    print(f"  Well ID: {well_id}")
    print(f"  Channel: {channel_idx}")
    print(f"  Resolution Level: 1 (default)")
    
    # Perform segmentation with timing
    print(f"\n🎯 Starting OME-Zarr segmentation...")
    
    start_time = time.time()
    try:
        # Call the segment_ome_zarr method
        result = await microsam_service.segment_ome_zarr(
            zarr_url=zarr_url,
            well_id=well_id,
            channel_idx=channel_idx,
            resolution_level=1,
            contrast_min_percentile=1.0,
            contrast_max_percentile=99.0,
            z_idx=0,
            t_idx=0
        )
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Segmentation completed!")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        # Display results
        print(f"\n📈 Results:")
        print(f"  Number of objects: {result['num_objects']}")
        print(f"  Pixel size: {result['pixel_size_um']:.6f} μm/pixel")
        print(f"  Image shape: {result['image_shape']}")
        print(f"  Resolution level: {result['resolution_level']}")
        print(f"  Preview saved: {result['preview_path']}")
        print(f"  Annotations saved: {result['json_path']}")
        
        # Show sample annotation
        if result['annotations'] and len(result['annotations']) > 0:
            print(f"\n📝 Sample Annotation (first object):")
            sample_ann = result['annotations'][0]
            print(f"  Object ID: {sample_ann['obj_id']}")
            print(f"  Type: {sample_ann['type']}")
            print(f"  Well: {sample_ann['well']}")
            print(f"  Channel: {sample_ann['channels'][0]['label']}")
            print(f"  Polygon WKT (first 100 chars): {sample_ann['polygon_wkt'][:100]}...")
            print(f"  Created: {sample_ann['created_at']}")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_ome_zarr_segmentation())

