from typing import Any, Dict, List

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field, BaseModel, ConfigDict
from ray import serve


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 2,  # Request 2 GPUs for training
        "memory": 12 * 1024 * 1024 * 1024,
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
        # Initialize checkpoint directory
        import os
        # Use absolute path to project's checkpoint directory
        # Since __file__ is not available in Ray Serve context, use a hardcoded path
        self.checkpoint_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/models/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_checkpoint = None
        
        # Initialize logs directory
        self.logs_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/logs"
        os.makedirs(self.logs_dir, exist_ok=True)
    
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
                imageio.imwrite(os.path.join(labels_dir, label_filename), mask)
        
        return temp_dir
    
    def _load_model(self, checkpoint_path=None):
        """Lazy load predictor and segmenter."""
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        
        # PERFORMANCE FIX: Avoid reloading if model is already loaded
        if self.predictor is not None and self.segmenter is not None:
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
        
        # Load predictor and segmenter (without tiling)
        self.predictor, self.segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=checkpoint_path,
            device=self.device,
            is_tiled=False,
        )
        
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
            self._load_model()
        
        # Log input image to logs folder
        self._save_image_to_logs(image, "input_image", subfolder="encoding")
        
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
            self._load_model()
        
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
    
    def _save_image_to_logs(self, image: np.ndarray, filename_prefix: str, subfolder: str = "") -> str:
        """Save image to logs folder with timestamp as JPEG.
        
        Args:
            image: Image array in any format (C, H, W), (H, W), or (H, W, C)
            filename_prefix: Prefix for the filename (e.g., 'input', 'segmentation', 'encoded')
            subfolder: Optional subfolder within logs (e.g., 'segmentation', 'encoding')
        
        Returns:
            Path to saved image file
        """
        import os
        from datetime import datetime
        from PIL import Image
        
        # Create subfolder if specified
        save_dir = os.path.join(self.logs_dir, subfolder) if subfolder else self.logs_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"{filename_prefix}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Convert image to 2D format for saving
        if len(image.shape) == 3:
            if image.shape[0] in [1, 3]:
                # (C, H, W) format
                if image.shape[0] == 1:
                    image_2d = image[0]  # (H, W)
                else:
                    image_2d = np.transpose(image, (1, 2, 0))  # (H, W, C)
            else:
                # Assume (H, W, C) format
                image_2d = image
        else:
            # Already 2D
            image_2d = image
        
        # Normalize to uint8 if needed
        if image_2d.dtype != np.uint8:
            if image_2d.max() <= 1.0:
                image_2d = (image_2d * 255).astype(np.uint8)
            elif image_2d.max() <= 255:
                image_2d = image_2d.astype(np.uint8)
            else:
                # Normalize to 0-255 range
                if image_2d.max() > image_2d.min():
                    image_2d = ((image_2d - image_2d.min()) / 
                               (image_2d.max() - image_2d.min()) * 255).astype(np.uint8)
                else:
                    image_2d = np.zeros_like(image_2d, dtype=np.uint8)
        
        # Save image as JPEG
        try:
            # Convert to PIL Image
            if len(image_2d.shape) == 2:
                # Grayscale: (H, W)
                pil_image = Image.fromarray(image_2d, mode='L')
            elif len(image_2d.shape) == 3:
                # RGB: (H, W, C)
                if image_2d.shape[2] == 3:
                    pil_image = Image.fromarray(image_2d, mode='RGB')
                elif image_2d.shape[2] == 1:
                    pil_image = Image.fromarray(image_2d[:, :, 0], mode='L')
                else:
                    pil_image = Image.fromarray(image_2d[:, :, :3], mode='RGB')
            else:
                pil_image = Image.fromarray(image_2d, mode='L')
            
            # Save as JPEG with 95% quality
            pil_image.save(filepath, format='JPEG', quality=95)
            return filepath
        except Exception as e:
            # Log error but don't fail the operation
            import logging
            logging.warning(f"Failed to save image to logs: {str(e)}")
            return ""
    
    def _combine_images_side_by_side(self, image1: np.ndarray, image2: np.ndarray, 
                                     label1: str = "Input", label2: str = "Segmentation") -> np.ndarray:
        """Combine two images side-by-side with labels for comparison.
        
        Args:
            image1: First image array
            image2: Second image array
            label1: Label for first image (default: "Input")
            label2: Label for second image (default: "Segmentation")
        
        Returns:
            Combined image as numpy array (H, W, C) in RGB format
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Normalize both images to (H, W) or (H, W, C) format
        def normalize_to_2d(img):
            if len(img.shape) == 3:
                if img.shape[0] in [1, 3]:
                    # (C, H, W) format
                    if img.shape[0] == 1:
                        return img[0]  # (H, W)
                    else:
                        return np.transpose(img, (1, 2, 0))  # (H, W, C)
            return img  # Already in correct format
        
        img1_2d = normalize_to_2d(image1)
        img2_2d = normalize_to_2d(image2)
        
        # Ensure both images have same height (resize if needed)
        h1, w1 = img1_2d.shape[:2]
        h2, w2 = img2_2d.shape[:2]
        
        # Use the maximum height
        max_h = max(h1, h2)
        
        # Resize images to same height if needed
        if h1 != max_h or h2 != max_h:
            from PIL import Image as PILImage
            if len(img1_2d.shape) == 2:
                pil1 = PILImage.fromarray(img1_2d, mode='L').resize((w1, max_h), PILImage.Resampling.LANCZOS)
            else:
                pil1 = PILImage.fromarray(img1_2d).resize((w1, max_h), PILImage.Resampling.LANCZOS)
            img1_2d = np.array(pil1)
            
            if len(img2_2d.shape) == 2:
                pil2 = PILImage.fromarray(img2_2d, mode='L').resize((w2, max_h), PILImage.Resampling.LANCZOS)
            else:
                pil2 = PILImage.fromarray(img2_2d).resize((w2, max_h), PILImage.Resampling.LANCZOS)
            img2_2d = np.array(pil2)
        
        # Normalize to uint8 for display
        def normalize_to_uint8(img):
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                elif img.max() <= 255:
                    img = img.astype(np.uint8)
                else:
                    if img.max() > img.min():
                        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
            return img
        
        img1_2d = normalize_to_uint8(img1_2d)
        img2_2d = normalize_to_uint8(img2_2d)
        
        # Convert grayscale to RGB if needed
        if len(img1_2d.shape) == 2:
            img1_rgb = np.stack([img1_2d] * 3, axis=-1)
        elif img1_2d.shape[2] == 1:
            img1_rgb = np.repeat(img1_2d, 3, axis=2)
        else:
            img1_rgb = img1_2d[:, :, :3]  # Take first 3 channels if more
        
        if len(img2_2d.shape) == 2:
            # For segmentation masks, use color mapping for better visibility
            img2_rgb = np.zeros((img2_2d.shape[0], img2_2d.shape[1], 3), dtype=np.uint8)
            # Create a colored overlay for the segmentation mask
            mask_normalized = img2_2d / 255.0 if img2_2d.max() > 0 else img2_2d
            # Use a color map (green overlay for segmentation)
            img2_rgb[:, :, 1] = (mask_normalized * 255).astype(np.uint8)  # Green channel
            # Also show original values in grayscale
            img2_rgb[:, :, 0] = img2_2d
            img2_rgb[:, :, 2] = img2_2d
        elif img2_2d.shape[2] == 1:
            mask_normalized = img2_2d[:, :, 0] / 255.0 if img2_2d.max() > 0 else img2_2d[:, :, 0]
            img2_rgb = np.zeros((img2_2d.shape[0], img2_2d.shape[1], 3), dtype=np.uint8)
            img2_rgb[:, :, 1] = (mask_normalized * 255).astype(np.uint8)
            img2_rgb[:, :, 0] = img2_2d[:, :, 0]
            img2_rgb[:, :, 2] = img2_2d[:, :, 0]
        else:
            img2_rgb = img2_2d[:, :, :3]
        
        # Combine images side-by-side
        combined_width = img1_rgb.shape[1] + img2_rgb.shape[1]
        combined_height = img1_rgb.shape[0]
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        combined[:, :img1_rgb.shape[1], :] = img1_rgb
        combined[:, img1_rgb.shape[1]:, :] = img2_rgb
        
        # Add labels using PIL
        pil_combined = Image.fromarray(combined, mode='RGB')
        draw = ImageDraw.Draw(pil_combined)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Add labels at the top of each image
        text_color = (255, 255, 0)  # Yellow text
        draw.text((10, 10), label1, fill=text_color, font=font)
        draw.text((img1_rgb.shape[1] + 10, 10), label2, fill=text_color, font=font)
        
        return np.array(pil_combined)

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
        
        # Load model if not already loaded
        if self.predictor is None or self.segmenter is None:
            self._load_model()
        
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
            
            # Convert to binary mask (uint8): 0 = background, 255 = segmented pixels
            # Convert from instance segmentation (uint32 with unique IDs) to simple binary mask
            if prediction.dtype != np.uint8 or prediction.max() > 1:
                # Create binary mask: anything > 0 becomes 255 (segmented), 0 stays as background
                prediction = (prediction > 0).astype(np.uint8) * 255
            
            # Create combined image (input + segmentation) for comparison
            combined_image = self._combine_images_side_by_side(
                image_array, 
                prediction, 
                label1="Input", 
                label2="Segmentation"
            )
            
            # Log combined image to logs folder (side-by-side comparison)
            self._save_image_to_logs(combined_image, "comparison", subfolder="segmentation")
            
            # Encode segmentation result to JPEG base64 string
            # STRICT: Always return JPEG base64, NO numpy arrays
            jpeg_base64_result = self._encode_segmentation_to_jpeg(prediction, quality=95)
            
            return jpeg_base64_result


if __name__ == "__main__":
    import asyncio

    micro_sam_trainer = MicroSamTrainer.func_or_class()

    async def test():
        response = await micro_sam_trainer.start_fit()
        print(response)

    asyncio.run(test())
