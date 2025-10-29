from typing import Any, Dict, List

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field, BaseModel, ConfigDict
from ray import serve
import requests
from shapely.geometry import Polygon
from shapely import wkt
from skimage import measure
from datetime import datetime
import json
import hashlib
import struct
import gzip
import asyncio
import aiohttp


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 2,  # Request 2 GPUs for training
        "memory": 14 * 1024 * 1024 * 1024,
        # "runtime_env": {"conda": ["microsam"]},  # Already running in the conda environment in terminal
    },
)
class MicroSamTrainer:
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self):
        # Training state
        self.fit_task = None
        self.fit_result = None
        self.fit_error = None
        self.fit_cancelled = False
        
        # Model state
        self.predictor = None
        self.segmenter = None
        self.model_type = "vit_b_lm"
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.current_is_tiled = None  # Track current tiling mode
        # Initialize checkpoint directory
        import os
        # Use absolute path to project's checkpoint directory
        # Since __file__ is not available in Ray Serve context, use a hardcoded path
        self.checkpoint_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/models/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_checkpoint = None
        
        # Initialize chunk cache directory for disk persistence
        self.chunk_cache_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/data/chunk_cache"
        os.makedirs(self.chunk_cache_dir, exist_ok=True)
    
    def _check_cuda(self):
        """Check CUDA availability without importing torch at module level."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _validate_image_format(self, image: np.ndarray) -> bool:
        """Validate image format for segmentation."""
        if not isinstance(image, np.ndarray):
            return False
        
        # Check shape: (C, H, W) where C in [1, 3]
        if len(image.shape) != 3:
            return False
        
        c, h, w = image.shape
        if c not in [1, 3] or h <= 0 or w <= 0:
            return False
        
        return True
    
    def _validate_coco_annotations(self, annotations: dict) -> bool:
        """Validate COCO format annotations."""
        required_keys = ['images', 'annotations', 'categories']
        
        # Check required top-level keys
        if not all(key in annotations for key in required_keys):
            return False
        
        # Check images structure
        if not isinstance(annotations['images'], list) or len(annotations['images']) == 0:
            return False
        
        # Check annotations structure
        if not isinstance(annotations['annotations'], list):
            return False
        
        # Check categories structure
        if not isinstance(annotations['categories'], list) or len(annotations['categories']) == 0:
            return False
        
        return True
    
    def _prepare_training_data(self, images: List[np.ndarray], annotations: dict, temp_dir: str) -> str:
        """Convert COCO format data to .tif files for training."""
        import os
        import json
        import imageio.v3 as imageio
        from scipy import ndimage
        
        # Create subdirectories
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create image ID to image mapping
        image_id_to_image = {img['id']: img for img in annotations['images']}
        
        # Create annotation ID to annotation mapping
        ann_id_to_ann = {ann['id']: ann for ann in annotations['annotations']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Process each image
        for i, image in enumerate(images):
            # Get corresponding image metadata
            if i >= len(annotations['images']):
                break
                
            img_meta = annotations['images'][i]
            image_id = img_meta['id']
            
            # Save image as .tif
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                # Convert (C, H, W) to (H, W) or (H, W, C)
                if image.shape[0] == 1:
                    image_2d = image[0]  # (H, W)
                else:
                    image_2d = np.transpose(image, (1, 2, 0))  # (H, W, C)
            else:
                image_2d = image  # Assume already (H, W)
            
            # Normalize to 0-255 if needed
            if image_2d.max() <= 1.0:
                image_2d = (image_2d * 255).astype(np.uint8)
            else:
                image_2d = image_2d.astype(np.uint8)
            
            # Save image
            img_filename = f"train_{i:03d}.tif"
            imageio.imwrite(os.path.join(images_dir, img_filename), image_2d)
            
            # Create segmentation mask
            if image_id in image_annotations:
                h, w = image_2d.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint16)
                
                for ann_idx, ann in enumerate(image_annotations[image_id]):
                    # Convert COCO segmentation to mask
                    if 'segmentation' in ann and ann['segmentation']:
                        # Handle polygon segmentation
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:  # At least 3 points
                                # Convert polygon to mask
                                from PIL import Image, ImageDraw
                                pil_mask = Image.new('L', (w, h), 0)
                                draw = ImageDraw.Draw(pil_mask)
                                
                                # Convert flat list to point pairs
                                points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                                draw.polygon(points, fill=ann_idx + 1)
                                
                                # Convert back to numpy
                                seg_mask = np.array(pil_mask)
                                mask[seg_mask > 0] = ann_idx + 1
                
                # Save label
                label_filename = f"train_{i:03d}.tif"
                imageio.imwrite(os.path.join(labels_dir, label_filename), mask)
        
        return temp_dir
    
    def _load_model(self, checkpoint_path=None, is_tiled=False):
        """Lazy load predictor and segmenter."""
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        
        # PERFORMANCE FIX: Avoid reloading if model is already loaded with same configuration
        if (self.predictor is not None and self.segmenter is not None and 
            self.current_is_tiled == is_tiled):
            return self.predictor, self.segmenter
        
        # Determine checkpoint to use
        if checkpoint_path is None:
            # Check for fine-tuned checkpoint
            import os
            best_checkpoint = os.path.join(self.checkpoint_dir, "best.pt")
            if os.path.exists(best_checkpoint):
                checkpoint_path = best_checkpoint
                self.current_checkpoint = checkpoint_path
            else:
                checkpoint_path = None  # Use pre-trained
                self.current_checkpoint = None
        
        # Load predictor and segmenter
        self.predictor, self.segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=checkpoint_path,
            device=self.device,
            is_tiled=is_tiled,
        )
        
        # Track current tiling mode
        self.current_is_tiled = is_tiled
        
        return self.predictor, self.segmenter

    def _fit_blocking(self, images: List[np.ndarray], annotations: dict, n_epochs: int):
        """Blocking training function to run in a thread."""
        import torch
        import os
        import tempfile
        import shutil
        import micro_sam.training as sam_training
        from torch_em.data import MinInstanceSampler
        
        try:
            # Create temporary directory for training data
            temp_dir = tempfile.mkdtemp(prefix="microsam_training_")
            
            # Prepare training data
            self._prepare_training_data(images, annotations, temp_dir)
            
            # Check for cancellation
            if self.fit_cancelled:
                self.fit_error = "Training cancelled by user"
                return
            
            # Create train/val split (80/20)
            images_dir = os.path.join(temp_dir, "images")
            labels_dir = os.path.join(temp_dir, "labels")
            
            # Get all image files
            import glob
            all_images = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
            all_labels = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))
            
            # Split into train/val
            split_idx = int(0.8 * len(all_images))
            train_images = all_images[:split_idx]
            train_labels = all_labels[:split_idx]
            val_images = all_images[split_idx:]
            val_labels = all_labels[split_idx:]
            
            # Rename files for train/val split
            for i, (img_path, label_path) in enumerate(zip(train_images, train_labels)):
                new_img_path = os.path.join(images_dir, f"train_{i:03d}.tif")
                new_label_path = os.path.join(labels_dir, f"train_{i:03d}.tif")
                os.rename(img_path, new_img_path)
                os.rename(label_path, new_label_path)
            
            for i, (img_path, label_path) in enumerate(zip(val_images, val_labels)):
                new_img_path = os.path.join(images_dir, f"valid_{i:03d}.tif")
                new_label_path = os.path.join(labels_dir, f"valid_{i:03d}.tif")
                os.rename(img_path, new_img_path)
                os.rename(label_path, new_label_path)
            
            # Check for cancellation
            if self.fit_cancelled:
                self.fit_error = "Training cancelled by user"
                return
            
            # Training configuration (hardcoded from training scripts)
            batch_size = 1  # Reduced for memory efficiency
            n_objects_per_batch = 4  # Reduced for memory efficiency
            patch_shape = (512, 512)
            learning_rate = 1e-4
            freeze_encoder = True
            early_stopping = 15
            scheduler_patience = 5
            
            # Use MinInstanceSampler
            sampler = MinInstanceSampler(min_size=20)
            
            # Create dataloaders
            train_loader = sam_training.default_sam_loader(
                raw_paths=images_dir,
                raw_key="train_*.tif",
                label_paths=labels_dir,
                label_key="train_*.tif",
                with_segmentation_decoder=True,
                patch_shape=(1, *patch_shape),
                batch_size=batch_size,
                shuffle=True,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )
            
            val_loader = sam_training.default_sam_loader(
                raw_paths=images_dir,
                raw_key="valid_*.tif",
                label_paths=labels_dir,
                label_key="valid_*.tif",
                with_segmentation_decoder=True,
                patch_shape=(1, *patch_shape),
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )
            
            # Check for cancellation
            if self.fit_cancelled:
                self.fit_error = "Training cancelled by user"
                return
            
            # Configure training parameters
            freeze_parts = ["image_encoder"] if freeze_encoder else None
            scheduler_kwargs = {
                "mode": "min",
                "factor": 0.8,
                "patience": scheduler_patience,
                "min_lr": 1e-7,
            }
            
            # Check for existing checkpoint to continue from
            latest_checkpoint = os.path.join(self.checkpoint_dir, "latest.pt")
            resume_from_checkpoint = latest_checkpoint if os.path.exists(latest_checkpoint) else None
            
            # Train the model
            checkpoint_name = "bioengine_shared"
            
            sam_training.train_sam(
                name=checkpoint_name,
                save_root="models",
                model_type=self.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                n_objects_per_batch=n_objects_per_batch,
                with_segmentation_decoder=True,
                freeze=freeze_parts,
                device=self.device,
                lr=learning_rate,
                early_stopping=early_stopping,
                scheduler_kwargs=scheduler_kwargs,
                save_every_kth_epoch=10,
                n_sub_iteration=6,
                mask_prob=0.6,
                checkpoint_path=resume_from_checkpoint,
                overwrite_training=False,
            )
            
            # Update current checkpoint
            best_checkpoint = os.path.join(self.checkpoint_dir, "best.pt")
            self.current_checkpoint = best_checkpoint
            
            self.fit_result = {
                "status": "completed",
                "checkpoint": best_checkpoint,
                "epochs": n_epochs,
                "message": "Training completed successfully"
            }
            
        except Exception as e:
            self.fit_error = f"Training failed: {str(e)}"
        finally:
            # Clean up temporary files
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _fit_background(self, images: List[np.ndarray], annotations: dict, n_epochs: int):
        """Async wrapper that runs blocking training in a thread."""
        import asyncio
        
        # Reset cancellation flag
        self.fit_cancelled = False
        
        # Enforce minimum 2 images requirement
        if len(images) < 2:
            raise ValueError(
                f"Minimum 2 images required for proper train/validation split. "
                f"Received {len(images)} image(s)."
            )
        
        # Run blocking training in a thread pool to avoid blocking the event loop
        # This allows health checks to continue responding
        try:
            await asyncio.to_thread(self._fit_blocking, images, annotations, n_epochs)
        except Exception as e:
            self.fit_error = f"Training failed: {str(e)}"

    @schema_method
    async def start_fit(
        self,
        images: List[Any] = Field(
            ...,
            description="List of training images as numpy arrays",
        ),
        annotations: Dict[str, Any] = Field(
            ...,
            description="COCO format annotations dictionary",
        ),
        n_epochs: int = Field(
            50,
            description="Number of epochs to train",
        ),
    ) -> Dict[str, str]:
        import asyncio
        
        # Validate inputs
        if not self._validate_coco_annotations(annotations):
            return {"status": "error", "message": "Invalid COCO annotations format"}
        
        for i, image in enumerate(images):
            if not self._validate_image_format(image):
                return {"status": "error", "message": f"Invalid image format at index {i}. Expected (C, H, W) where C in [1, 3]"}
        
        # Start training task
        self.fit_task = asyncio.create_task(self._fit_background(images, annotations, n_epochs))
        self.fit_result = None
        self.fit_error = None

        return {"status": "started", "message": "Fit task started in background"}

    @schema_method
    async def get_fit_status(
        self,
    ) -> Dict[str, str]:
        if self.fit_task is None:
            return {
                "status": "not_started",
                "message": "Fit task has not been started.",
            }
        elif self.fit_task.done():
            if self.fit_error is not None:
                return {"status": "error", "message": str(self.fit_error)}
            else:
                return {
                    "status": "completed",
                    "message": "Fit task completed successfully.",
                }
        else:
            return {"status": "running", "message": "Fit task is still running."}

    @schema_method
    async def cancel_fit(
        self,
    ) -> Dict[str, str]:
        if self.fit_task is None:
            return {
                "status": "not_started",
                "message": "Fit task has not been started.",
            }
        elif self.fit_task.done():
            return {"status": "completed", "message": "Fit task has already completed."}

        # Set cancellation flag - this is a cooperative cancellation signal
        self.fit_cancelled = True

        # Cancel the asyncio task
        self.fit_task.cancel()

        # Wait for the task to actually finish (with timeout)
        try:
            await asyncio.wait_for(self.fit_task, timeout=5.0)
        except asyncio.TimeoutError:
            # Task didn't finish in time, but flag is set so it will check when it completes
            pass
        except asyncio.CancelledError:
            # Expected when task is cancelled
            pass

        self.fit_task = None
        self.fit_result = None
        self.fit_error = "Task was cancelled by user"

        return {"status": "cancelled", "message": "Fit task has been cancelled."}

    @schema_method
    async def encode_image(
        self,
        image: Any = Field(
            ...,
            description="Input data as numpy array of shape (C, H, W)",
        ),
    ) -> Any:
        # Validate input format
        if not self._validate_image_format(image):
            raise ValueError(f"Invalid image format. Expected (C, H, W) where C in [1, 3], got {image.shape}")
        
        # Load model if not already loaded
        if self.predictor is None:
            self._load_model(is_tiled=False)  # Encoding doesn't need tiling
        
        # Convert image to the format expected by SAM
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:
            # Convert (C, H, W) to (H, W) for grayscale or (H, W, C) for RGB
            if image.shape[0] == 1:
                image_2d = image[0]  # (H, W)
            else:
                image_2d = np.transpose(image, (1, 2, 0))  # (H, W, C)
        else:
            image_2d = image  # Assume already (H, W)
        
        # Set image in predictor to compute embeddings
        self.predictor.set_image(image_2d)
        
        # Extract embeddings from predictor's internal state
        # The predictor stores the image features internally
        # We need to access the features that were computed
        if hasattr(self.predictor, 'features') and self.predictor.features is not None:
            # Extract the image features (embeddings)
            features = self.predictor.features
            if hasattr(features, 'cpu'):
                features = features.cpu().numpy()
            
            # Flatten to 1D array
            embedding = features.flatten()
            return embedding
        else:
            raise RuntimeError("Failed to extract embeddings from predictor")

    @schema_method
    async def download_mask_decoder(
        self,
    ) -> bytes:
        import torch
        import io
        
        # Load model if not already loaded
        if self.predictor is None or self.segmenter is None:
            self._load_model(is_tiled=False)  # Download doesn't need tiling
        
        # Get the current checkpoint path
        checkpoint_path = self.current_checkpoint
        if checkpoint_path is None:
            # Use pre-trained model
            from micro_sam.util import get_sam_model
            model = get_sam_model(model_type=self.model_type, device=self.device)
        else:
            # Load fine-tuned model
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            from micro_sam.util import get_sam_model
            model = get_sam_model(model_type=self.model_type, device=self.device)
            model.load_state_dict(checkpoint["model_state"])
        
        # Extract mask decoder state dict
        mask_decoder_state = {}
        for name, param in model.mask_decoder.named_parameters():
            mask_decoder_state[name] = param.cpu()
        
        for name, buffer in model.mask_decoder.named_buffers():
            mask_decoder_state[name] = buffer.cpu()
        
        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(mask_decoder_state, buffer)
        buffer.seek(0)
        
        return buffer.getvalue()

    def _decode_jpeg_image(self, jpeg_base64: str) -> np.ndarray:
        """Decode base64-encoded JPEG image to numpy array."""
        import base64
        import io
        from PIL import Image
        
        try:
            # Decode base64 to get JPEG bytes
            jpeg_bytes = base64.b64decode(jpeg_base64)
            
            # Decode JPEG bytes to PIL Image
            pil_image = Image.open(io.BytesIO(jpeg_bytes))
            
            # Convert PIL Image to numpy array
            image_array = np.array(pil_image)
            
            # Handle different image formats
            if len(image_array.shape) == 2:
                # Grayscale: (H, W) -> (1, H, W)
                image_array = np.expand_dims(image_array, axis=0)
            elif len(image_array.shape) == 3:
                # RGB: (H, W, C) -> (C, H, W)
                image_array = np.transpose(image_array, (2, 0, 1))
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to decode JPEG image: {str(e)}")
    
    def _encode_segmentation_to_jpeg(self, segmentation: np.ndarray, quality: int = 95) -> str:
        """Encode segmentation mask to base64-encoded JPEG string."""
        import base64
        import io
        from PIL import Image
        
        try:
            # Normalize segmentation to 0-255 uint8
            if segmentation.dtype != np.uint8:
                # Normalize to 0-255 range
                if segmentation.max() > segmentation.min():
                    seg_normalized = ((segmentation - segmentation.min()) / 
                                    (segmentation.max() - segmentation.min()) * 255).astype(np.uint8)
                else:
                    seg_normalized = np.zeros_like(segmentation, dtype=np.uint8)
            else:
                seg_normalized = segmentation
            
            # Handle different segmentation formats
            if len(seg_normalized.shape) == 2:
                # Grayscale mask: (H, W)
                pil_image = Image.fromarray(seg_normalized, mode='L')
            elif len(seg_normalized.shape) == 3:
                # If (C, H, W), convert to (H, W) for grayscale
                if seg_normalized.shape[0] == 1:
                    pil_image = Image.fromarray(seg_normalized[0], mode='L')
                else:
                    # RGB: (H, W, C)
                    pil_image = Image.fromarray(seg_normalized, mode='RGB')
            else:
                raise ValueError(f"Unsupported segmentation shape: {seg_normalized.shape}")
            
            # Encode to JPEG
            jpeg_buffer = io.BytesIO()
            pil_image.save(jpeg_buffer, format='JPEG', quality=quality)
            jpeg_bytes = jpeg_buffer.getvalue()
            
            # Encode to base64
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            return jpeg_base64
            
        except Exception as e:
            raise ValueError(f"Failed to encode segmentation to JPEG: {str(e)}")

    def _fetch_zarr_metadata(self, zarr_url: str) -> dict:
        """Fetch zarr metadata from .zattrs file."""
        try:
            # Ensure URL ends with /
            if not zarr_url.endswith('/'):
                zarr_url += '/'
            
            # Fetch .zattrs metadata
            zattrs_url = f"{zarr_url}.zattrs"
            response = requests.get(zattrs_url, timeout=30)
            response.raise_for_status()
            
            metadata = response.json()
            return metadata
            
        except Exception as e:
            raise ValueError(f"Failed to fetch zarr metadata from {zattrs_url}: {str(e)}")
    
    def _list_available_chunks(self, zarr_url: str, resolution_level: int) -> list:
        """List available chunks at a given resolution level."""
        try:
            # Ensure URL ends with /
            if not zarr_url.endswith('/'):
                zarr_url += '/'
            
            # List chunks at resolution level
            chunks_url = f"{zarr_url}{resolution_level}/"
            response = requests.get(chunks_url, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            files = response.json()
            
            # Extract chunk filenames and parse coordinates
            chunks = []
            for file_info in files:
                if file_info['type'] == 'file' and file_info['name'] not in ['.zarray', '.zattrs', '.zgroup']:
                    # Parse chunk coordinates from filename: t.c.z.y.x
                    parts = file_info['name'].split('.')
                    if len(parts) == 5:
                        t, c, z, y, x = map(int, parts)
                        chunks.append({
                            'name': file_info['name'],
                            't': t,
                            'c': c,
                            'z': z,
                            'y': y,
                            'x': x
                        })
            
            return chunks
            
        except Exception as e:
            raise ValueError(f"Failed to list chunks from {chunks_url}: {str(e)}")
    
    def _fetch_zarr_chunk(self, chunk_url: str, chunk_shape: tuple, dtype: str) -> np.ndarray:
        """Fetch and decode a single zarr chunk."""
        try:
            # Fetch raw chunk data
            response = requests.get(chunk_url, timeout=30)
            response.raise_for_status()
            
            # Get raw bytes
            raw_data = response.content
            
            # Try to decompress if gzip compressed
            try:
                decompressed_data = gzip.decompress(raw_data)
            except:
                # If decompression fails, assume it's not compressed
                decompressed_data = raw_data
            
            # Decode binary data to numpy array
            chunk_array = np.frombuffer(decompressed_data, dtype=dtype)
            
            # Reshape to chunk shape
            chunk_array = chunk_array.reshape(chunk_shape)
            
            return chunk_array
            
        except Exception as e:
            raise ValueError(f"Failed to fetch chunk from {chunk_url}: {str(e)}")
    
    def _get_chunk_cache_path(self, chunk_url: str) -> str:
        """Generate cache file path for a chunk URL."""
        import os
        # Create hash-based filename from chunk URL for uniqueness
        cache_key = hashlib.md5(chunk_url.encode('utf-8')).hexdigest()
        # Use subdirectory structure (first 2 chars) to avoid too many files in one dir
        subdir = cache_key[:2]
        subdir_path = os.path.join(self.chunk_cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        cache_path = os.path.join(subdir_path, f"{cache_key}.npz")
        return cache_path
    
    async def _load_chunk_from_cache(self, chunk_url: str) -> np.ndarray:
        """Load chunk from disk cache if available."""
        import os
        cache_path = self._get_chunk_cache_path(chunk_url)
        if os.path.exists(cache_path):
            try:
                # Load asynchronously in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                chunk_data = await loop.run_in_executor(None, np.load, cache_path)
                chunk_array = chunk_data['chunk']
                chunk_data.close()  # Close the npz file
                return chunk_array
            except Exception as e:
                # If cache load fails, remove corrupted cache file
                print(f"  Warning: Failed to load cache for chunk, removing corrupted file: {str(e)}")
                try:
                    os.remove(cache_path)
                except:
                    pass
                return None
        return None
    
    async def _save_chunk_to_cache(self, chunk_url: str, chunk_array: np.ndarray):
        """Save chunk to disk cache."""
        import os
        cache_path = self._get_chunk_cache_path(chunk_url)
        try:
            # Save asynchronously in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: np.savez_compressed(cache_path, chunk=chunk_array)
            )
        except Exception as e:
            # Cache save failures shouldn't break the workflow
            print(f"  Warning: Failed to save chunk to cache: {str(e)}")
    
    async def _compose_image_from_chunks(
        self, 
        zarr_url: str, 
        resolution_level: int, 
        channel_idx: int, 
        z_idx: int, 
        t_idx: int,
        chunk_info: dict
    ) -> np.ndarray:
        """Compose full 2D image from zarr chunks using parallel fetching."""
        try:
            # Ensure URL ends with /
            if not zarr_url.endswith('/'):
                zarr_url += '/'
            
            # Get chunk metadata
            zarray_url = f"{zarr_url}{resolution_level}/.zarray"
            response = requests.get(zarray_url, timeout=30)
            response.raise_for_status()
            zarray = response.json()
            
            dtype = zarray['dtype']
            if dtype.startswith('<') or dtype.startswith('>'):
                dtype = dtype[1:]  # Remove endianness prefix
            chunk_shape = tuple(zarray['chunks'])
            
            # List available chunks
            all_chunks = self._list_available_chunks(zarr_url, resolution_level)
            
            # Filter chunks matching t, c, z
            matching_chunks = [
                ch for ch in all_chunks 
                if ch['t'] == t_idx and ch['c'] == channel_idx and ch['z'] == z_idx
            ]
            
            if not matching_chunks:
                raise ValueError(f"No chunks found for t={t_idx}, c={channel_idx}, z={z_idx}")
            
            # Determine full image dimensions
            max_y = max(ch['y'] for ch in matching_chunks)
            max_x = max(ch['x'] for ch in matching_chunks)
            
            # Chunk shape is (t, c, z, y, x) - we want the last 2 dimensions
            chunk_h, chunk_w = chunk_shape[-2], chunk_shape[-1]
            
            # Initialize output array
            full_h = (max_y + 1) * chunk_h
            full_w = (max_x + 1) * chunk_w
            output = np.zeros((full_h, full_w), dtype=dtype)
            
            # Fetch chunks in parallel using aiohttp with disk caching
            async def fetch_chunk(session, chunk):
                chunk_url = f"{zarr_url}{resolution_level}/{chunk['name']}"
                try:
                    # Try loading from cache first
                    cached_chunk_2d = await self._load_chunk_from_cache(chunk_url)
                    if cached_chunk_2d is not None:
                        return chunk, cached_chunk_2d
                    
                    # Cache miss - fetch from network
                    async with session.get(chunk_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        raw_data = await resp.read()
                        
                        # Try to decompress if gzip compressed
                        try:
                            decompressed_data = gzip.decompress(raw_data)
                        except:
                            decompressed_data = raw_data
                        
                        # Decode binary data to numpy array
                        chunk_array = np.frombuffer(decompressed_data, dtype=dtype)
                        
                        # Reshape: chunk is (t, c, z, y, x) but we want (y, x)
                        chunk_array = chunk_array.reshape(chunk_shape)
                        # Extract the 2D slice (last 2 dimensions)
                        chunk_2d = chunk_array[0, 0, 0, :, :]
                        
                        # Save to cache asynchronously (don't await to avoid blocking)
                        # Use create_task to run in background
                        asyncio.create_task(self._save_chunk_to_cache(chunk_url, chunk_2d))
                        
                        return chunk, chunk_2d
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    print(f"Warning: Failed to fetch chunk {chunk['name']}: {error_type}: {error_msg}")
                    return chunk, None
            
            # Fetch chunks in batches to avoid overwhelming the server
            batch_size = 10  # Fetch 10 chunks at a time
            results = []
            
            async with aiohttp.ClientSession() as session:
                # Process chunks in batches
                total_batches = (len(matching_chunks) + batch_size - 1) // batch_size
                for i in range(0, len(matching_chunks), batch_size):
                    batch = matching_chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                    
                    # Create tasks for this batch
                    tasks = [fetch_chunk(session, chunk) for chunk in batch]
                    
                    # Process chunks as they complete (allows yielding control for health checks)
                    # This prevents blocking the event loop and allows health checks to be processed
                    batch_results_map = {}
                    for coro in asyncio.as_completed(tasks):
                        try:
                            chunk, chunk_data = await coro
                            batch_results_map[chunk['name']] = (chunk, chunk_data)
                        except Exception as e:
                            # Extract chunk from exception if possible, otherwise use placeholder
                            pass
                        # Yield control after each chunk to allow health checks
                        await asyncio.sleep(0)
                    
                    # Reconstruct batch results in original order
                    batch_results = []
                    for chunk in batch:
                        if chunk['name'] in batch_results_map:
                            batch_results.append(batch_results_map[chunk['name']])
                        else:
                            # Chunk failed or wasn't processed
                            batch_results.append((chunk, None))
                    
                    results.extend(batch_results)
                    
                    # Small delay between batches to avoid overwhelming the server and allow health checks
                    await asyncio.sleep(0.2)
            
            # Place chunks in output array
            successful = 0
            failed = 0
            for chunk, chunk_data in results:
                if chunk_data is not None:
                    y_start = chunk['y'] * chunk_h
                    y_end = y_start + chunk_h
                    x_start = chunk['x'] * chunk_w
                    x_end = x_start + chunk_w
                    
                    output[y_start:y_end, x_start:x_end] = chunk_data
                    successful += 1
                else:
                    failed += 1
            
            print(f"  Chunk loading complete: {successful} succeeded, {failed} failed out of {len(matching_chunks)} total")
            
            return output
            
        except Exception as e:
            raise ValueError(f"Failed to compose image from chunks: {str(e)}")
    
    async def _apply_contrast_adjustment(
        self, 
        image: np.ndarray, 
        min_percentile: float, 
        max_percentile: float
    ) -> np.ndarray:
        """Apply contrast adjustment using min/max clipping and normalization.
        
        For uint8 images (0-255 range): Simple min/max clipping and linear normalization.
        Memory-efficient: processes in blocks to avoid large float64 allocations.
        
        Note: min_percentile and max_percentile parameters are kept for API compatibility but not used.
        """
        try:
            h, w = image.shape[:2]
            image_dtype = image.dtype
            
            # Step 1: Compute min/max on downsampled image for speed (for very large images)
            max_sample_size = 2048  # Target max dimension for sampling
            loop = asyncio.get_event_loop()
            
            def compute_min_max(img):
                """Compute min and max values."""
                return float(img.min()), float(img.max())
            
            if h > max_sample_size or w > max_sample_size:
                # Downsample using decimation (taking every nth pixel) for speed
                scale_factor = max(h, w) / max_sample_size
                step = max(1, int(scale_factor))
                image_sample = image[::step, ::step]
                print(f"  Computing min/max on downsampled image: {image_sample.shape} (from {image.shape})")
                vmin, vmax = await loop.run_in_executor(None, compute_min_max, image_sample)
                
                # Yield control before computing full image min/max
                await asyncio.sleep(0)
                
                # Use full image min/max to ensure we don't miss actual range
                # This is still CPU-intensive, so run in executor
                img_min, img_max = await loop.run_in_executor(
                    None, 
                    lambda: (float(image.min()), float(image.max()))
                )
                
                # Use the more conservative range (wider)
                vmin = min(vmin, img_min)
                vmax = max(vmax, img_max)
                
                print(f"  Min/Max: vmin={vmin:.2f}, vmax={vmax:.2f} (image range: {img_min}-{img_max}, dtype={image_dtype})")
            else:
                # Image is small enough, process directly
                vmin, vmax = await loop.run_in_executor(None, compute_min_max, image)
                img_min, img_max = vmin, vmax
                print(f"  Min/Max: vmin={vmin:.2f}, vmax={vmax:.2f} (image range: {img_min}-{img_max}, dtype={image_dtype})")
            
            # Check if image is constant
            if vmin == vmax:
                print(f"  Image is constant ({vmin}), returning unchanged")
                return image.astype(np.uint8)
            
            # Yield control to allow health checks
            await asyncio.sleep(0)
            
            # Step 2: Apply adjustment in blocks (clip to min/max range, then normalize to 0-255)
            # Pre-allocate output as uint8 to avoid large float64 arrays
            image_normalized = np.zeros_like(image, dtype=np.uint8)
            
            # Process in blocks to avoid memory issues
            block_size = 4096  # Process 4K rows at a time
            total_blocks = (h + block_size - 1) // block_size
            
            def process_block(start_row):
                """Clip block to min/max range and normalize to 0-255."""
                end_row = min(start_row + block_size, h)
                block = image[start_row:end_row, :]
                
                # Step 1: Clip to min/max range
                block_clipped = np.clip(block, vmin, vmax)
                
                # Step 2: Normalize to 0-255: (value - vmin) / (vmax - vmin) * 255
                # Use float32 for intermediate calculations to save memory vs float64
                range_size = vmax - vmin
                if range_size > 0:
                    block_normalized = ((block_clipped.astype(np.float32) - vmin) / range_size * 255).astype(np.uint8)
                else:
                    block_normalized = block_clipped.astype(np.uint8)
                
                return start_row, end_row, block_normalized
            
            # Process blocks in parallel batches
            # Reduce parallel blocks to yield more frequently for health checks
            parallel_blocks = 2  # Process 2 blocks in parallel (reduced for more frequent yields)
            
            print(f"  Processing {total_blocks} blocks ({parallel_blocks} at a time)...")
            
            for block_idx in range(0, total_blocks, parallel_blocks):
                # Get the actual row indices for this batch of blocks
                block_starts = [min(block_idx + i, total_blocks - 1) * block_size 
                               for i in range(parallel_blocks) 
                               if (block_idx + i) < total_blocks]
                
                # Process batch in parallel
                tasks = [loop.run_in_executor(None, process_block, start) 
                        for start in block_starts]
                results = await asyncio.gather(*tasks)
                
                # Place results in output array
                for start_row, end_row, block_data in results:
                    image_normalized[start_row:end_row, :] = block_data
                
                # Yield control after every batch to allow health checks
                await asyncio.sleep(0)
                
                # Progress update every few batches
                if block_idx % (parallel_blocks * 3) == 0 or block_idx == 0:
                    completed = min(block_idx + parallel_blocks, total_blocks)
                    progress = completed / total_blocks * 100
                    print(f"  Contrast adjustment progress: {progress:.1f}% ({completed}/{total_blocks} blocks)")
            
            return image_normalized
            
        except Exception as e:
            raise ValueError(f"Failed to apply contrast adjustment: {str(e)}")
    
    def _get_pixel_size_um(self, metadata: dict, resolution_level: int) -> float:
        """Extract pixel size in micrometers from zarr metadata."""
        try:
            # Navigate to the correct dataset
            multiscales = metadata['multiscales'][0]
            datasets = multiscales['datasets']
            
            if resolution_level >= len(datasets):
                raise ValueError(f"Resolution level {resolution_level} not found in metadata")
            
            # Get scale transformation
            dataset = datasets[resolution_level]
            transformations = dataset['coordinateTransformations']
            
            # Find scale transformation
            for transform in transformations:
                if transform['type'] == 'scale':
                    # Scale is [t, c, z, y, x] - we want y or x (should be same)
                    scale = transform['scale']
                    # Return y scale (index -2) or x scale (index -1)
                    pixel_size_um = scale[-1]  # x scale
                    return pixel_size_um
            
            raise ValueError("Scale transformation not found in metadata")
            
        except Exception as e:
            raise ValueError(f"Failed to extract pixel size from metadata: {str(e)}")
    
    def _instance_mask_to_polygons(
        self, 
        mask: np.ndarray, 
        pixel_size_um: float, 
        image_shape: tuple,
        metadata: dict,
        well_id: str,
        channel_idx: int,
        resolution_level: int,
        contrast_percentiles: list
    ) -> list:
        """Convert instance segmentation mask to polygon annotations in WKT format."""
        try:
            annotations = []
            
            # Get unique instance IDs (excluding background 0)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]
            
            # Image center (well center)
            center_y_pixel = image_shape[0] / 2
            center_x_pixel = image_shape[1] / 2
            
            # Get channel label from metadata
            channel_label = "Unknown"
            if 'omero' in metadata and 'channels' in metadata['omero']:
                channels = metadata['omero']['channels']
                if channel_idx < len(channels):
                    channel_label = channels[channel_idx].get('label', f"Channel {channel_idx}")
            
            # Extract dataset ID and name from metadata or URL
            dataset_name = metadata.get('squid_canvas', {}).get('fileset_name', 'unknown')
            dataset_id = "unknown"
            
            # Process each instance
            for inst_id in instance_ids:
                # Create binary mask for this instance
                instance_mask = (mask == inst_id).astype(np.uint8)
                
                # Find contours
                contours = measure.find_contours(instance_mask, 0.5)
                
                if len(contours) == 0:
                    continue
                
                # Take the longest contour
                contour = max(contours, key=len)
                
                # Convert contour to polygon coordinates
                # contour is (y, x) in pixel coordinates
                coords = []
                for y_pixel, x_pixel in contour:
                    # Convert to mm relative to well center
                    x_mm = (x_pixel - center_x_pixel) * pixel_size_um / 1000.0
                    y_mm = (y_pixel - center_y_pixel) * pixel_size_um / 1000.0
                    coords.append((x_mm, y_mm))
                
                # Create shapely polygon
                try:
                    poly = Polygon(coords)
                    
                    # Simplify polygon to reduce vertex count
                    poly_simplified = poly.simplify(tolerance=2.0 * pixel_size_um / 1000.0, preserve_topology=True)
                    
                    # Convert to WKT
                    polygon_wkt = poly_simplified.wkt
                    
                    # Generate unique object ID
                    timestamp = int(datetime.now().timestamp())
                    hash_str = hashlib.md5(f"{inst_id}_{timestamp}".encode()).hexdigest()[:8]
                    obj_id = f"obj_{timestamp}_{hash_str}"
                    
                    # Create annotation dictionary
                    annotation = {
                        "obj_id": obj_id,
                        "well": well_id,
                        "type": "polygon",
                        "description": "Automated cell segmentation",
                        "timestamp": timestamp,
                        "created_at": datetime.now().isoformat(),
                        "bbox": None,
                        "polygon_wkt": polygon_wkt,
                        "channels": [{"index": channel_idx, "label": channel_label}],
                        "process_settings": {
                            "model": self.model_type,
                            "resolution_level": resolution_level,
                            "pixel_size_um": pixel_size_um,
                            "contrast_percentiles": contrast_percentiles
                        },
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "embeddings": None
                    }
                    
                    annotations.append(annotation)
                    
                except Exception as e:
                    print(f"Warning: Failed to create polygon for instance {inst_id}: {e}")
                    continue
            
            return annotations
            
        except Exception as e:
            raise ValueError(f"Failed to convert mask to polygons: {str(e)}")

    @schema_method
    async def segment_all(
        self,
        image_or_embedding: Any = Field(
            ...,
            description="Base64-encoded JPEG string (required for images). Numpy arrays are NOT accepted for images.",
        ),
        embedding: bool = Field(
            False,
            description="If True, the input is treated as an embedding; otherwise, as a JPEG image.",
        ),
        tile_shape: Any = Field(
            None,
            description="Tile shape for large images (e.g., (1024, 1024)). None for auto-detection.",
        ),
        halo: Any = Field(
            None,
            description="Tile overlap for large images (e.g., (256, 256)). None for auto-detection.",
        ),
    ) -> Any:
        from micro_sam.automatic_segmentation import automatic_instance_segmentation
        
        # Input validation and decoding
        if embedding:
            # For embeddings, expect 1D array
            if not isinstance(image_or_embedding, np.ndarray):
                image_or_embedding = np.array(image_or_embedding)
            if len(image_or_embedding.shape) != 1:
                raise ValueError(f"Expected 1D embedding array, got shape {image_or_embedding.shape}")
        else:
            # STRICT: For images, ONLY accept base64-encoded JPEG string
            # NO numpy arrays, NO fallback, NO exceptions
            if not isinstance(image_or_embedding, str):
                raise ValueError(
                    f"ONLY base64-encoded JPEG strings are accepted for images. "
                    f"Received {type(image_or_embedding).__name__}. "
                    f"Numpy arrays and other formats are NOT supported. "
                    f"Please compress your image to JPEG and encode as base64 string."
                )
            
            # Additional check: ensure it's not an empty string
            if len(image_or_embedding.strip()) == 0:
                raise ValueError("Empty string provided. Must be a valid base64-encoded JPEG string.")
            
            # Decode JPEG image from base64
            image_array = self._decode_jpeg_image(image_or_embedding)
            
            # Validate decoded image format
            if not self._validate_image_format(image_array):
                raise ValueError(f"Invalid decoded image format. Expected (C, H, W) where C in [1, 3], got {image_array.shape}")
        
        # Determine if tiling is needed
        needs_tiling = False
        if tile_shape is not None and halo is not None:
            needs_tiling = True
        elif tile_shape is None and halo is None and not embedding:
            # Auto-detect if tiling is needed for large images
            # CRITICAL FIX: Increased threshold to avoid unnecessary tiling
            if len(image_array.shape) >= 2:
                # Check image dimensions (H, W from (C, H, W))
                h, w = image_array.shape[1], image_array.shape[2]
                if h > 4096 or w > 4096:
                    needs_tiling = True
                    # CRITICAL FIX: Larger tiles with smaller overlap for better performance
                    tile_shape = (2048, 2048)  # Increased from 1024x1024
                    halo = (128, 128)          # Reduced from 256x256
        
        # Save image if tiling is needed
        if needs_tiling:
            import os
            import imageio.v3 as imageio
            from datetime import datetime
            
            # Create data directory if it doesn't exist
            data_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert image to saveable format
            # image_array is (C, H, W), convert to (H, W) or (H, W, C)
            if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
                if image_array.shape[0] == 1:
                    image_to_save = image_array[0]  # (H, W) for grayscale
                else:
                    image_to_save = np.transpose(image_array, (1, 2, 0))  # (H, W, C) for RGB
            else:
                image_to_save = image_array  # Assume already (H, W)
            
            # Normalize to 0-255 if needed
            if image_to_save.max() <= 1.0:
                image_to_save = (image_to_save * 255).astype(np.uint8)
            else:
                image_to_save = image_to_save.astype(np.uint8)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"tiled_image_{timestamp}.tif"
            image_path = os.path.join(data_dir, image_filename)
            
            # Save image
            imageio.imwrite(image_path, image_to_save)
        
        # Load model if not already loaded or if tiling requirements changed
        if (self.predictor is None or self.segmenter is None or 
            self.current_is_tiled != needs_tiling):
            self._load_model(is_tiled=needs_tiling)
        
        # Run automatic instance segmentation
        if embedding:
            # For embeddings, we need to handle this differently
            # This is a simplified approach - in practice, you might need to reconstruct the image
            # or use the embedding directly with the segmenter
            raise NotImplementedError("Embedding-based segmentation not yet implemented")
        else:
            # Convert image to the format expected by microSAM
            if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
                # Convert (C, H, W) to (H, W) for grayscale or (H, W, C) for RGB
                if image_array.shape[0] == 1:
                    image_2d = image_array[0]  # (H, W)
                else:
                    image_2d = np.transpose(image_array, (1, 2, 0))  # (H, W, C)
            else:
                image_2d = image_array  # Assume already (H, W)
            
            # Prepare kwargs for segmentation
            kwargs = {
                "predictor": self.predictor,
                "segmenter": self.segmenter,
                "input_path": image_2d,
                "ndim": 2,
            }
            
            # Add tiling parameters if provided
            if tile_shape is not None and halo is not None:
                kwargs["tile_shape"] = tile_shape
                kwargs["halo"] = halo
                # CRITICAL FIX: Increase batch size for better GPU utilization
                kwargs["batch_size"] = 4  # Increased from 1 for better performance
            
            # Run segmentation
            prediction = automatic_instance_segmentation(**kwargs)
            
            # Convert prediction to numpy array if it's not already
            if not isinstance(prediction, np.ndarray):
                # If prediction is a dict or other format, extract the mask
                if isinstance(prediction, dict):
                    # Try common keys for segmentation masks
                    for key in ['segmentation', 'mask', 'masks', 'result']:
                        if key in prediction:
                            prediction = prediction[key]
                            break
                    else:
                        raise ValueError(f"Could not find segmentation mask in prediction dict. Keys: {list(prediction.keys())}")
                else:
                    prediction = np.array(prediction)
            
            # Encode segmentation result to JPEG base64 string
            # STRICT: Always return JPEG base64, NO numpy arrays
            jpeg_base64_result = self._encode_segmentation_to_jpeg(prediction, quality=95)
            
            return jpeg_base64_result

    @schema_method
    async def segment_ome_zarr(
        self,
        zarr_url: str = Field(
            ...,
            description="Base URL to OME-Zarr data (e.g., https://.../data.zarr/)",
        ),
        well_id: str = Field(
            ...,
            description="Well identifier (e.g., 'C3')",
        ),
        channel_idx: int = Field(
            ...,
            description="Channel index to segment",
        ),
        contrast_min_percentile: float = Field(
            1.0,
            description="Minimum contrast percentile for normalization",
        ),
        contrast_max_percentile: float = Field(
            99.0,
            description="Maximum contrast percentile for normalization",
        ),
        resolution_level: int = Field(
            1,
            description="Pyramid resolution level (default 1 for ~1.25μm/pixel)",
        ),
        z_idx: int = Field(
            0,
            description="Z-slice index",
        ),
        t_idx: int = Field(
            0,
            description="Timepoint index",
        ),
        tile_shape: Any = Field(
            None,
            description="Tile shape for micro-sam segmentation (e.g., [1024, 1024]). Defaults to [1024, 1024] if None (recommended by micro-sam for large images).",
        ),
        halo: Any = Field(
            None,
            description="Tile overlap for segmentation (e.g., [256, 256]). Defaults to [256, 256] if None (recommended: larger than half max object radius).",
        ),
    ) -> Dict[str, Any]:
        """Segment OME-Zarr data and return polygon annotations in WKT format with mm coordinates."""
        
        try:
            print(f"🔬 Starting OME-Zarr segmentation for {zarr_url}")
            
            # 1. Fetch zarr metadata first (needed for dimensions)
            print("📊 Fetching zarr metadata...")
            metadata = self._fetch_zarr_metadata(zarr_url)
            
            # Set optimal tile size based on actual image dimensions
            # List chunks to estimate image size before composition
            temp_chunks = self._list_available_chunks(zarr_url, resolution_level)
            if temp_chunks:
                # Estimate dimensions from chunk coordinates
                max_y = max(ch['y'] for ch in temp_chunks) + 1
                max_x = max(ch['x'] for ch in temp_chunks) + 1
                # Get chunk size from .zarray
                zarray_url = f"{zarr_url}{resolution_level}/.zarray"
                zarray_response = requests.get(zarray_url, timeout=30)
                if zarray_response.ok:
                    zarray = zarray_response.json()
                    chunk_shape = tuple(zarray['chunks'])
                    estimated_h = max_y * chunk_shape[-2]
                    estimated_w = max_x * chunk_shape[-1]
                    
                    # Choose tile size to keep total tiles reasonable (<1000 tiles)
                    # Formula: target_tiles = (est_h * est_w) / (tile_size^2)
                    # For <1000 tiles: tile_size = sqrt((est_h * est_w) / 1000)
                    estimated_pixels = estimated_h * estimated_w
                    optimal_tile_size = int(np.sqrt(estimated_pixels / 800))  # Target ~800 tiles
                    # Round to nearest power of 2 for efficiency (2048, 4096, 8192, etc.)
                    optimal_tile_size = 2 ** int(np.log2(optimal_tile_size))
                    # Clamp between 2048 and 8192
                    optimal_tile_size = max(2048, min(8192, optimal_tile_size))
                else:
                    optimal_tile_size = 4096  # Default for very large images
            else:
                optimal_tile_size = 4096  # Default
            
            # Set defaults if None (to avoid JSON serialization issues with tuple/list defaults)
            if tile_shape is None:
                tile_shape = [optimal_tile_size, optimal_tile_size]
                print(f"  Auto-selected tile size: {optimal_tile_size}x{optimal_tile_size} (target: <1000 tiles)")
            if halo is None:
                # Halo proportional to tile size: ~12.5% overlap
                # This balances accuracy vs computation
                halo_size = max(256, optimal_tile_size // 8)  # At least 256, or 1/8 of tile
                halo = [halo_size, halo_size]
                print(f"  Auto-selected halo: {halo_size}x{halo_size} (~{100*halo_size/optimal_tile_size:.1f}% overlap)")
            
            # 2. Compose image from chunks
            print(f"🧩 Composing image from chunks (resolution level {resolution_level}, channel {channel_idx})...")
            image = await self._compose_image_from_chunks(
                zarr_url=zarr_url,
                resolution_level=resolution_level,
                channel_idx=channel_idx,
                z_idx=z_idx,
                t_idx=t_idx,
                chunk_info={}
            )
            
            print(f"✅ Image composed: shape {image.shape}, dtype {image.dtype}")
            
            # # 3. Apply contrast adjustment
            # print(f"🎨 Applying contrast adjustment ({contrast_min_percentile}%-{contrast_max_percentile}%)...")
            # image_adjusted = await self._apply_contrast_adjustment(
            #     image=image,
            #     min_percentile=contrast_min_percentile,
            #     max_percentile=contrast_max_percentile
            # )
            
            # 4. Convert to (C, H, W) format for micro-sam
            # Add channel dimension for grayscale
            image_chw = np.expand_dims(image, axis=0)  # (1, H, W)
            
            print(f"🖼️ Image prepared for segmentation: shape {image_chw.shape}")
            
            # 5. Run segmentation directly using automatic_instance_segmentation
            # micro-sam accepts numpy arrays directly - no file saving needed!
            # This avoids TIF file size limits and JPEG encoding issues for very large images
            print("🎯 Running micro-sam segmentation with tiling...")
            from micro_sam.automatic_segmentation import automatic_instance_segmentation
            
            # Convert (1, H, W) to (H, W) - micro-sam expects 2D or 3D (RGB)
            image_2d = image_chw[0]
            
            # Convert lists to tuples if needed
            tile_shape_tuple = tuple(tile_shape) if isinstance(tile_shape, list) else tile_shape
            halo_tuple = tuple(halo) if isinstance(halo, list) else halo
            
            # Load model with tiling enabled (always use for such large images)
            # micro-sam FAQ recommends: tile_shape=(1024, 1024), halo=(256, 256)
            needs_tiling = True  # Always use tiling for images this large
            if self.predictor is None or self.segmenter is None or not self.current_is_tiled:
                self._load_model(is_tiled=needs_tiling)
            
            # Run segmentation with tiling - pass numpy array directly!
            kwargs = {
                "predictor": self.predictor,
                "segmenter": self.segmenter,
                "input_path": image_2d,  # Pass numpy array directly, not file path!
                "ndim": 2,
            }
            
            if tile_shape_tuple is not None and halo_tuple is not None:
                kwargs["tile_shape"] = tile_shape_tuple
                kwargs["halo"] = halo_tuple
                kwargs["batch_size"] = 4  # Process multiple tiles in parallel
            
            seg_array = automatic_instance_segmentation(**kwargs)
            
            # Ensure seg_array is numpy array
            if not isinstance(seg_array, np.ndarray):
                seg_array = np.array(seg_array)
            
            print(f"✅ Segmentation completed: shape {seg_array.shape}")
            
            # 8. Get pixel size from metadata
            pixel_size_um = self._get_pixel_size_um(metadata, resolution_level)
            print(f"📏 Pixel size: {pixel_size_um:.6f} μm/pixel")
            
            # 9. Convert instance mask to polygons
            print("🔺 Converting masks to polygon annotations...")
            annotations = self._instance_mask_to_polygons(
                mask=seg_array,
                pixel_size_um=pixel_size_um,
                image_shape=image.shape,
                metadata=metadata,
                well_id=well_id,
                channel_idx=channel_idx,
                resolution_level=resolution_level,
                contrast_percentiles=[contrast_min_percentile, contrast_max_percentile]
            )
            
            print(f"✅ Generated {len(annotations)} polygon annotations")
            
            # 10. Save preview JPEG and JSON files locally
            import os
            import imageio.v3 as imageio
            
            # Create output directory
            output_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/data/segmentation_previews"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_filename = f"segmentation_{well_id}_{timestamp_str}.jpg"
            json_filename = f"annotations_{well_id}_{timestamp_str}.json"
            
            preview_path = os.path.join(output_dir, preview_filename)
            json_path = os.path.join(output_dir, json_filename)
            
            # Save preview image
            print(f"💾 Saving preview to {preview_path}...")
            imageio.imwrite(preview_path, seg_array)
            
            # Save annotations JSON
            print(f"💾 Saving annotations to {json_path}...")
            with open(json_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            # 11. Return result
            result = {
                "annotations": annotations,
                "preview_path": preview_path,
                "json_path": json_path,
                "num_objects": len(annotations),
                "pixel_size_um": pixel_size_um,
                "image_shape": image.shape,
                "resolution_level": resolution_level
            }
            
            print(f"🎉 Segmentation completed successfully!")
            
            return result
            
        except Exception as e:
            error_msg = f"OME-Zarr segmentation failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise ValueError(error_msg)
    
    def _encode_image_to_jpeg_base64(self, image: np.ndarray, quality: int = 85) -> str:
        """Encode numpy image to base64-encoded JPEG string."""
        import base64
        import io
        from PIL import Image
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            pil_image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Encode to JPEG
        jpeg_buffer = io.BytesIO()
        pil_image.save(jpeg_buffer, format='JPEG', quality=quality)
        jpeg_bytes = jpeg_buffer.getvalue()
        
        # Encode to base64
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        return jpeg_base64


if __name__ == "__main__":
    import asyncio

    micro_sam_trainer = MicroSamTrainer.func_or_class()

    async def test():
        response = await micro_sam_trainer.start_fit()
        print(response)

    asyncio.run(test())
