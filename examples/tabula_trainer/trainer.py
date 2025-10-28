import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

pip_requirements = [
    "aiohttp==3.9.0",
    "aiosignal==1.3.1",
    "anyio==3.7.1",
    "attrs==23.1.0",
    "certifi==2023.11.17",
    "charset-normalizer==3.3.2",
    "click==8.1.7",
    "einops==0.7.0",
    "fastapi==0.106.0",
    "filelock==3.13.1",
    "flwr==1.22.0",  # Directly used
    "frozenlist==1.4.0",
    "fsspec==2023.10.0",
    "h11==0.14.0",
    "idna==3.4",
    "Jinja2==3.1.2",
    "joblib==1.3.2",
    "lightning-utilities==0.10.0",
    "MarkupSafe==2.1.3",
    "multidict==6.0.4",
    "networkx==3.2.1",
    "ninja==1.11.1.1",
    "numpy==1.26.0",  # Directly used
    "packaging==22.0",
    "protobuf==4.25.5",
    "psutil==5.9.6",
    "Pygments==2.17.2",
    "pytorch-lightning==2.2.0",  # Directly used
    "pyyaml==6.0.2",  # Directly used
    "requests==2.32.3",  # Directly used
    "scikit-learn==1.3.0",  # Directly used
    "scipy==1.10.1",
    "six==1.16.0",
    "sniffio==1.3.0",
    "threadpoolctl==3.2.0",
    "torch==1.13.1",  # Directly used
    # pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    "torchmetrics==1.2.0",
    "tqdm==4.67.1",
    "typing_extensions==4.12.2",
    "urllib3==1.26.18",
    "wrapt==1.16.0",
    "zarr==3.1.3",  # Directly used
]

if os.getenv("ENABLE_FLASH_ATTENTION") == "1":
    # Optional performance packages
    pip_requirements.extend(["flash-attn==2.3.5 --no-build-isolation"])


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {"pip": pip_requirements},
    },
    max_ongoing_requests=1,  # Important to guarantee thread-safety
    max_queued_requests=5,
    autoscaling_config={
        "min_replicas": 0,
        "initial_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 0.8,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 300,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=120.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class TabulaTrainer:
    def __init__(
        self,
        datasets: List[str],
        client_id: Optional[str] = None,
        client_name: Optional[str] = None,
        initial_weights: Optional[str] = None,
    ):
        """
        Flower Client for Federated Learning
        """
        import torch

        from tabula.training import ModelConfig
        from bioengine.datasets import BioEngineDatasets

        self.bioengine_datasets: (
            BioEngineDatasets  # Automatically injected by BioEngine
        )

        # Store dataset names
        self.datasets = datasets
        self.training_datasets = {}

        # Set client_id and client_name
        self.client_id = client_id or str(uuid.uuid4())
        self.client_name = client_name or self.client_id

        # Load model configuration and initial weights
        config_path = self._download_from_artifact("framework.yaml")
        self.config = ModelConfig(config_path)

        self.weights_path = (
            self._download_from_artifact(initial_weights) if initial_weights else None
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === BioEngine App Method - will be called when the deployment is started ===

    async def async_init(self):
        from tabula.distributed import FederatedClient

        # Initialize executor for running blocking operations
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Initialize task tracking (no lock needed with max_ongoing_requests=1)
        self.fit_task: Optional[asyncio.Task] = None
        self.evaluate_task: Optional[asyncio.Task] = None
        self.fit_result: Optional[Tuple[List[np.ndarray], int, Dict[str, float]]] = None
        self.evaluate_result: Optional[Tuple[float, int, Dict[str, float]]] = None
        self.fit_error: Optional[str] = None
        self.evaluate_error: Optional[str] = None

        # Cancellation flags for cooperative cancellation
        self.fit_cancelled: bool = False
        self.evaluate_cancelled: bool = False

        # Get available zarr files from BioEngine datasets
        zarr_sources = []
        for dataset_name in self.datasets:
            files = await self.bioengine_datasets.list_files(dataset_name)
            for file_name in files:
                if file_name.endswith(".zarr"):
                    zarr_store = await self.bioengine_datasets.get_file(
                        dataset_name=dataset_name, file_name=file_name
                    )
                    zarr_sources.append(zarr_store)

        # Get information about the training datasets
        available_datasets = await self.bioengine_datasets.list_datasets()
        self.training_datasets = {
            dataset_name: dataset_info
            for dataset_name, dataset_info in available_datasets.items()
            if dataset_name in self.datasets
        }

        # Create federated client
        self.client = FederatedClient(
            client_id=self.client_id,
            client_name=self.client_name,
            config=self.config,
            zarr_sources=zarr_sources,
            initial_weights=self.weights_path,
            device=self.device,
            should_cancel_fit=lambda: self.fit_cancelled,
            should_cancel_evaluate=lambda: self.evaluate_cancelled,
        )

    async def test_deployment(self):
        pass

    # === Internal Methods ===

    def _download_from_artifact(self, file_path: str) -> Path:
        """
        Download a file from a Hypha artifact if not running locally.

        # TODO: add artifact manager to application for private artifacts
        """
        import requests

        local_artifact_path = os.environ.get("BIOENGINE_LOCAL_ARTIFACT_PATH")
        if local_artifact_path:
            local_file_path = (
                Path(local_artifact_path).resolve() / "tabula_trainer" / file_path
            )

        else:
            local_file_path = Path(file_path).resolve()

            # Download framework.yaml from Hypha if not running locally
            hypha_server_url = os.environ["HYPHA_SERVER_URL"]
            hypha_artifact_id = os.environ["HYPHA_ARTIFACT_ID"]
            artifact_workspace, artifact_alias = hypha_artifact_id.split(":")

            url = f"{hypha_server_url}/{artifact_workspace}/artifacts/{artifact_alias}/files/{file_path}"

            response = requests.get(url)
            # Raise an error for bad responses
            response.raise_for_status()
            # Save content to local file
            local_file_path.write_bytes(response.content)

        return local_file_path

    async def _fit_background(self, parameters: List[np.ndarray]):
        """Background task for fit operation."""
        try:
            # Reset cancellation flag
            self.fit_cancelled = False

            # Run fit in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor, self.client.fit, parameters
            )

            # Check if cancelled during execution
            if self.fit_cancelled:
                self.fit_result = None
                self.fit_error = "Task was cancelled"
                self.fit_task = None
                return None

            # Update the result (no lock needed with max_ongoing_requests=1)
            # Query client for total batches at the end of execution and store it
            try:
                total_batches = self.client.get_fit_total_batches()
            except Exception:
                total_batches = np.nan
            # Store result as (parameters, num_samples, metrics, total_batches)
            self.fit_result = (result[0], result[1], result[2], total_batches)
            self.fit_error = None
            self.fit_task = None
            return result
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            self.fit_result = None
            self.fit_error = "Task was cancelled"
            self.fit_task = None
            raise
        except Exception as e:
            # Handle errors
            self.fit_result = None
            self.fit_error = str(e)
            self.fit_task = None
            raise

    async def _evaluate_background(self, parameters: List[np.ndarray]):
        """Background task for evaluate operation."""
        try:
            # Reset cancellation flag
            self.evaluate_cancelled = False

            # Run evaluate in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor, self.client.evaluate, parameters
            )

            # Check if cancelled during execution
            if self.evaluate_cancelled:
                self.evaluate_result = None
                self.evaluate_error = "Task was cancelled"
                self.evaluate_task = None
                return None

            # Update the result (no lock needed with max_ongoing_requests=1)
            try:
                total_batches = self.client.get_evaluate_total_batches()
            except Exception:
                total_batches = np.nan
            # Store result as (loss, num_samples, metrics, total_batches)
            self.evaluate_result = (result[0], result[1], result[2], total_batches)
            self.evaluate_error = None
            self.evaluate_task = None
            return result
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            self.evaluate_result = None
            self.evaluate_error = "Task was cancelled"
            self.evaluate_task = None
            raise
        except Exception as e:
            # Handle errors
            self.evaluate_result = None
            self.evaluate_error = str(e)
            self.evaluate_task = None
            raise

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===

    @schema_method
    async def list_datasets(self) -> Dict[str, dict]:
        """List training datasets."""
        return self.training_datasets

    @schema_method
    async def get_properties(self) -> Dict[str, str | int]:
        """Get training and validation metrics."""
        client_properties = self.client.get_properties()
        client_properties["artifact_id"] = os.environ["HYPHA_ARTIFACT_ID"]
        return client_properties

    # TODO: check why arbitrary_types_allowed is needed for numpy arrays
    @schema_method(arbitrary_types_allowed=True)
    async def get_parameters(self) -> Dict[str, Union[List[np.ndarray], str]]:
        """Get the current model parameters. Disabled during fit operations."""
        # Check if fit is running (parameters shouldn't be read during training)
        if self.fit_task is not None and not self.fit_task.done():
            raise RuntimeError("Cannot get parameters while fit task is running")

        # Evaluate doesn't modify parameters, so it's safe to get them during evaluation
        return self.client.get_parameters()

    @schema_method(arbitrary_types_allowed=True)
    async def start_fit(
        self,
        parameters: List[np.ndarray] = Field(
            ..., description="A list of NumPy arrays containing the model weights"
        ),
    ) -> Dict[str, str]:
        """Start fitting the model to the training data with the given parameters."""
        # Check if any task is already running
        if self.fit_task is not None and not self.fit_task.done():
            return {"status": "error", "message": "A fit task is already running"}

        if self.evaluate_task is not None and not self.evaluate_task.done():
            return {
                "status": "error",
                "message": "An evaluate task is already running. Only one task (fit or evaluate) can run at a time.",
            }

        # Start background task
        self.fit_task = asyncio.create_task(self._fit_background(parameters))
        self.fit_result = None
        self.fit_error = None

        return {"status": "started", "message": "Fit task started in background"}

    @schema_method(arbitrary_types_allowed=True)
    async def start_evaluate(
        self,
        parameters: List[np.ndarray] = Field(
            ..., description="A list of NumPy arrays containing the model weights"
        ),
    ) -> Dict[str, str]:
        """Start evaluating the model on the test data with the given parameters."""

        # Check if any task is already running
        if self.evaluate_task is not None and not self.evaluate_task.done():
            return {"status": "error", "message": "An evaluate task is already running"}

        if self.fit_task is not None and not self.fit_task.done():
            return {
                "status": "error",
                "message": "A fit task is already running. Only one task (fit or evaluate) can run at a time.",
            }

        # Start background task
        self.evaluate_task = asyncio.create_task(self._evaluate_background(parameters))
        self.evaluate_result = None
        self.evaluate_error = None

        return {"status": "started", "message": "Evaluate task started in background"}

    @schema_method
    async def get_fit_result(self) -> Dict[str, Union[dict, str, int, float]]:
        """Get the result of the fit task."""
        if self.fit_task is None:
            if self.fit_result is not None:
                # Task completed successfully
                params, num_samples, metrics, total_batches = self.fit_result
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "current_batch": total_batches,
                    "total_batches": total_batches,
                    "result": {
                        "parameters": params,
                        "num_samples": num_samples,
                        "metrics": metrics,
                    },
                }
            elif self.fit_error is not None:
                # Task failed
                return {"status": "failed", "error": self.fit_error}
            else:
                # No task has been started
                return {
                    "status": "not_started",
                    "message": "No fit task has been started",
                }

        # Task is still running
        if self.fit_task.done():
            if self.fit_task.exception() is not None:
                return {"status": "failed", "error": str(self.fit_task.exception())}
            # Should have result now
            if self.fit_result is not None:
                # Task finished while background task still referenced
                params, num_samples, metrics, total_batches = self.fit_result
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "current_batch": total_batches,
                    "total_batches": total_batches,
                    "result": {
                        "parameters": params,
                        "num_samples": num_samples,
                        "metrics": metrics,
                    },
                }

        # Get current progress
        progress_info = self.client.get_fit_progress()
        return {
            "status": "running",
            "message": "Fit task is still running",
            "progress": progress_info["progress"],
            "current_batch": progress_info["current_batch"],
            "total_batches": progress_info["total_batches"],
        }

    @schema_method
    async def get_evaluate_result(self) -> Dict[str, Union[dict, str, int, float]]:
        """Get the result of the evaluate task."""
        if self.evaluate_task is None:
            if self.evaluate_result is not None:
                # Task completed successfully
                loss, num_samples, metrics, total_batches = self.evaluate_result
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "current_batch": total_batches,
                    "total_batches": total_batches,
                    "result": {
                        "loss": loss,
                        "num_samples": num_samples,
                        "metrics": metrics,
                    },
                }
            elif self.evaluate_error is not None:
                # Task failed
                return {"status": "failed", "error": self.evaluate_error}
            else:
                # No task has been started
                return {
                    "status": "not_started",
                    "message": "No evaluate task has been started",
                }

        # Task is still running
        if self.evaluate_task.done():
            if self.evaluate_task.exception() is not None:
                return {
                    "status": "failed",
                    "error": str(self.evaluate_task.exception()),
                }
            # Should have result now
            if self.evaluate_result is not None:
                # Task finished while background task still referenced - use stored total_batches
                loss, num_samples, metrics, total_batches = self.evaluate_result
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "current_batch": total_batches,
                    "total_batches": total_batches,
                    "result": {
                        "loss": loss,
                        "num_samples": num_samples,
                        "metrics": metrics,
                    },
                }

        # Get current progress
        progress_info = self.client.get_evaluate_progress()
        return {
            "status": "running",
            "message": "Evaluate task is still running",
            "progress": progress_info["progress"],
            "current_batch": progress_info["current_batch"],
            "total_batches": progress_info["total_batches"],
        }

    @schema_method
    async def cancel_fit(self) -> Dict[str, str]:
        """Cancel the ongoing fit task."""
        if self.fit_task is None:
            return {"status": "no_task", "message": "No fit task is running"}

        if self.fit_task.done():
            return {
                "status": "already_completed",
                "message": "Fit task already completed",
            }

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

        return {
            "status": "cancelled",
            "message": "Fit task cancellation requested. Note: PyTorch Lightning training may continue until the current epoch completes.",
        }

    @schema_method
    async def cancel_evaluate(self) -> Dict[str, str]:
        """Cancel the ongoing evaluate task."""
        if self.evaluate_task is None:
            return {"status": "no_task", "message": "No evaluate task is running"}

        if self.evaluate_task.done():
            return {
                "status": "already_completed",
                "message": "Evaluate task already completed",
            }

        # Set cancellation flag - this is a cooperative cancellation signal
        self.evaluate_cancelled = True

        # Cancel the asyncio task
        self.evaluate_task.cancel()

        # Wait for the task to actually finish (with timeout)
        try:
            await asyncio.wait_for(self.evaluate_task, timeout=5.0)
        except asyncio.TimeoutError:
            # Task didn't finish in time, but flag is set so it will check when it completes
            pass
        except asyncio.CancelledError:
            # Expected when task is cancelled
            pass

        self.evaluate_task = None
        self.evaluate_result = None
        self.evaluate_error = "Task was cancelled by user"

        return {
            "status": "cancelled",
            "message": "Evaluate task cancellation requested. Note: PyTorch Lightning evaluation may continue until completion.",
        }


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from bioengine.datasets import BioEngineDatasets

    print("=== Testing TabulaTrainer ===")

    # Set working directory
    app_workdir = Path.home() / ".bioengine" / "apps" / "tabula-trainer"
    app_workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(app_workdir)
    print(f"Current working directory: {os.getcwd()}")

    # Set app directory
    local_artifact_dir = Path(__file__).parent.parent
    os.environ["BIOENGINE_LOCAL_ARTIFACT_PATH"] = str(local_artifact_dir.resolve())
    print(f"Local artifact path: {os.environ['BIOENGINE_LOCAL_ARTIFACT_PATH']}")

    # Set artifact ID
    os.environ["HYPHA_ARTIFACT_ID"] = "chiron-platform/tabula-trainer"

    async def test_trainer():
        # Initialize a BioEngineDatasets instance
        bioengine_datasets = BioEngineDatasets(
            hypha_token=os.getenv("HYPHA_TOKEN"),
        )

        available_datasets = await bioengine_datasets.list_datasets()
        print(f"Available datasets: {available_datasets}")

        # Create TabulaTrainer instance
        trainer = TabulaTrainer.func_or_class(
            datasets=list(available_datasets.keys()),
            client_id="test_client",
            client_name="Test Client",
            initial_weights="initial_weights/lung.pth",
        )
        trainer.bioengine_datasets = bioengine_datasets
        await trainer.async_init()

        print("Listing datasets:")
        datasets = await trainer.list_datasets({})
        print(datasets)

        print("Getting properties:")
        properties = await trainer.get_properties({})
        print(properties)

        print("Getting parameters:")
        parameters = await trainer.get_parameters({})
        print(type(parameters))

        print("Starting fit:")
        fit_start = await trainer.start_fit(parameters, {})
        print(fit_start)

        print("Polling fit result:")
        await asyncio.sleep(5.0)
        fit_result = await trainer.get_fit_result({})
        print(fit_result)

        await asyncio.sleep(10.0)
        fit_result = await trainer.get_fit_result({})
        print(fit_result)

        print("Cancelling fit:")
        cancel_result = await trainer.cancel_fit({})
        print(cancel_result)

        await asyncio.sleep(5.0)
        fit_result = await trainer.get_fit_result({})
        print(fit_result)

        print("Starting evaluate:")
        eval_start = await trainer.start_evaluate(parameters, {})
        print(eval_start)

        print("Polling evaluate result:")
        await asyncio.sleep(3.0)
        eval_result = await trainer.get_evaluate_result({})
        print(eval_result)

        print("Cancelling evaluate:")
        cancel_eval = await trainer.cancel_evaluate({})
        print(cancel_eval)

        await asyncio.sleep(5.0)
        eval_result = await trainer.get_evaluate_result({})
        print(eval_result)

    asyncio.run(test_trainer())
