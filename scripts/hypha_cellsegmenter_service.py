#!/usr/bin/env python3
"""
Pure Hypha Cell Segmentation Service - No BioEngine/Ray required.

This service runs directly as a Hypha service without BioEngine or Ray Serve.
It provides cell segmentation using microSAM and Cellpose models.

Usage:
    conda activate microsam
    python scripts/hypha_cellsegmenter_service.py
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CellSegmenterHypha:
    """Pure Hypha cell segmentation service - no Ray/BioEngine dependencies."""
    
    def __init__(self):
        # Training state
        self.fit_task = None
        self.fit_result = None
        self.fit_error = None
        self.fit_cancelled = False
        
        # Model state - microSAM
        self.predictor = None
        self.segmenter = None
        self.model_type = "vit_b_lm"
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.is_tiled = False
        
        # Model state - Cellpose
        self.cellpose_model = None
        
        # Initialize checkpoint directory
        self.checkpoint_dir = str(project_root / "models" / "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_checkpoint = None
        
        print(f"✅ CellSegmenter initialized on device: {self.device}")
    
    def _check_cuda(self):
        """Check CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _validate_image_format(self, image: np.ndarray) -> bool:
        """Validate image format for segmentation."""
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) != 3:
            return False
        c, h, w = image.shape
        if c not in [1, 3] or h <= 0 or w <= 0:
            return False
        return True
    
    def _validate_coco_annotations(self, annotations: dict) -> bool:
        """Validate COCO format annotations."""
        required_keys = ['images', 'annotations', 'categories']
        if not all(key in annotations for key in required_keys):
            return False
        if not isinstance(annotations['images'], list) or len(annotations['images']) == 0:
            return False
        if not isinstance(annotations['annotations'], list):
            return False
        if not isinstance(annotations['categories'], list) or len(annotations['categories']) == 0:
            return False
        return True
    
    def _load_model(self, checkpoint_path=None, is_tiled=False):
        """Lazy load predictor and segmenter for microSAM."""
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        
        if (self.predictor is not None and self.segmenter is not None and 
            self.is_tiled == is_tiled):
            return self.predictor, self.segmenter
        
        if checkpoint_path is None:
            best_checkpoint = os.path.join(self.checkpoint_dir, "best.pt")
            if os.path.exists(best_checkpoint):
                checkpoint_path = best_checkpoint
                self.current_checkpoint = checkpoint_path
            else:
                checkpoint_path = None
                self.current_checkpoint = None
        
        self.predictor, self.segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=checkpoint_path,
            device=self.device,
            is_tiled=is_tiled,
        )
        self.is_tiled = is_tiled
        
        return self.predictor, self.segmenter
    
    def _load_cellpose_model(self):
        """Lazy load Cellpose model (cyto3)."""
        from cellpose import models
        
        if self.cellpose_model is None:
            self.cellpose_model = models.CellposeModel(gpu=(self.device == "cuda"), model_type='cyto3')
            print("Cellpose cyto3 model loaded")
        
        return self.cellpose_model
    
    def _segment_with_cellpose(self, image_2d: np.ndarray) -> np.ndarray:
        """Run Cellpose segmentation on image."""
        self._load_cellpose_model()
        
        masks, flows, styles = self.cellpose_model.eval(
            [image_2d],
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            batch_size=1,
        )
        
        return masks[0]
    
    def _decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64-encoded PNG image to numpy array."""
        import base64
        import io
        from PIL import Image
        
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image, dtype=np.uint8)
        
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, :, :]
        elif len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        
        return image_array
    
    @schema_method
    async def ping(self) -> str:
        """Health check endpoint."""
        return "pong"
    
    @schema_method
    async def get_status(self) -> Dict[str, str]:
        """Get service status."""
        return {
            "status": "running",
            "device": self.device,
            "model_type": self.model_type,
            "checkpoint": self.current_checkpoint or "pre-trained",
        }
    
    @schema_method
    async def segment_all(
        self,
        image: Any = Field(..., description="Base64-encoded PNG string"),
        method: str = Field("cellpose", description="'microsam' or 'cellpose'"),
        tiling: bool = Field(False, description="Use tiling for large images (microSAM only)"),
        min_cell_size: int = Field(100, description="Minimum cell size in pixels"),
    ) -> Dict[str, Any]:
        """
        Run cell segmentation on an image.
        
        Returns:
            Dictionary with 'success', 'mask', and 'message' keys.
            The 'mask' is a base64-encoded PNG string.
        """
        import base64
        import io
        from PIL import Image
        
        try:
            # Validate method
            if method not in ["microsam", "cellpose"]:
                return {"success": False, "message": f"Invalid method: {method}"}
            
            # Decode image
            if isinstance(image, str):
                image_array = self._decode_image(image)
            else:
                image_array = np.array(image)
            
            if not self._validate_image_format(image_array):
                return {"success": False, "message": f"Invalid image format: {image_array.shape}"}
            
            # Convert to 2D
            if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
                if image_array.shape[0] == 1:
                    image_2d = image_array[0]
                else:
                    image_2d = np.transpose(image_array, (1, 2, 0))
            else:
                image_2d = image_array
            
            # Ensure uint8
            if image_2d.dtype != np.uint8:
                if image_2d.max() <= 1.0:
                    image_2d = (image_2d * 255).astype(np.uint8)
                else:
                    image_2d = image_2d.astype(np.uint8)
            
            # Run segmentation
            if method == "cellpose":
                instances = await asyncio.to_thread(self._segment_with_cellpose, image_2d)
            else:  # microsam
                from micro_sam.automatic_segmentation import automatic_instance_segmentation
                
                await asyncio.to_thread(self._load_model, is_tiled=tiling)
                
                kwargs = {
                    "predictor": self.predictor,
                    "segmenter": self.segmenter,
                    "input_path": image_2d,
                    "ndim": 2,
                    "verbose": False,
                    "min_size": min_cell_size,
                }
                
                if tiling:
                    kwargs["tile_shape"] = (512, 512)
                    kwargs["halo"] = (128, 128)
                
                instances = await asyncio.to_thread(automatic_instance_segmentation, **kwargs)
            
            # Encode result to PNG
            if instances.dtype != np.uint8:
                if instances.max() > instances.min():
                    seg_normalized = ((instances - instances.min()) / 
                                    (instances.max() - instances.min()) * 255).astype(np.uint8)
                else:
                    seg_normalized = np.zeros_like(instances, dtype=np.uint8)
            else:
                seg_normalized = instances
            
            pil_image = Image.fromarray(seg_normalized, mode='L')
            png_buffer = io.BytesIO()
            pil_image.save(png_buffer, format='PNG')
            png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "mask": png_base64,
                "message": f"Segmentation complete. Found {len(np.unique(instances)) - 1} cells.",
                "cell_count": int(len(np.unique(instances)) - 1),
            }
            
        except Exception as e:
            return {"success": False, "message": f"Segmentation failed: {str(e)}"}


async def main():
    """Start the Hypha service."""
    load_dotenv()
    
    # Get token from environment
    token = os.getenv("HYPHA_TOKEN")
    if not token:
        print("❌ HYPHA_TOKEN not found in .env file")
        print("Please add your Hypha token to .env: HYPHA_TOKEN=your_token_here")
        sys.exit(1)
    
    server_url = "https://hypha.aicell.io"
    workspace = "reef-imaging"
    service_id = "cell-segmenter-hypha"
    
    print(f"🔌 Connecting to Hypha server: {server_url}")
    print(f"📁 Workspace: {workspace}")
    print(f"🔧 Service ID: {service_id}")
    
    try:
        server = await connect_to_server({
            "server_url": server_url,
            "token": token,
            "workspace": workspace,
        })
        
        print("✅ Connected to Hypha server")
        
        # Create service instance
        segmenter = CellSegmenterHypha()
        
        # Register service
        service = await server.register_service({
            "name": "Cell Segmenter (Pure Hypha)",
            "id": service_id,
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            # Expose methods
            "ping": segmenter.ping,
            "get_status": segmenter.get_status,
            "segment_all": segmenter.segment_all,
        })
        
        print(f"✅ Service registered!")
        print(f"   Service ID: {service.id}")
        print(f"   Full ID: {server.config.workspace}/{service_id}")
        print(f"\n🚀 Service is running. Press Ctrl+C to stop.")
        
        # Keep service running
        await server.serve()
        
    except asyncio.CancelledError:
        print("\n👋 Service stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Service stopped")
