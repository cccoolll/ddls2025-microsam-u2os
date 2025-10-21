from typing import Any, Dict

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {"conda": ["microsam"]},  # Name of conda environment
    },
)
class MicroSamTrainer:
    def __init__(self):
        self.fit_task = None
        self.fit_result = None
        self.fit_error = None
        self.fit_cancelled = False

    async def _fit_background(self, n_epochs: int):
        import torch

        # Reset cancellation flag
        self.fit_cancelled = False

        pass

    @schema_method
    async def start_fit(
        self,
        n_epochs: int = Field(
            1,
            description="Number of epochs to train",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, str]:
        self.fit_task = asyncio.create_task(self._fit_background(n_epochs=n_epochs))
        self.fit_result = None
        self.fit_error = None

        return {"status": "started", "message": "Fit task started in background"}

    @schema_method
    async def get_fit_status(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
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
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
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

    @schema_method(arbitrary_types_allowed=True)  # needed for numpy array
    async def encode_image(
        self,
        image: np.ndarray = Field(
            ...,
            description="Input data as numpy array of shape (C, H, W)",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> np.ndarray:
        pass

    @schema_method
    async def download_mask_decoder(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> bytes:
        pass

    @schema_method(arbitrary_types_allowed=True)  # needed for numpy array
    async def segment_all(
        self,
        image_or_embedding: np.ndarray = Field(
            ...,
            description="Input data as numpy array of shape (C, H, W) if image, or (D,) if embedding.",
        ),
        embedding: bool = Field(
            False,
            description="If True, the input is treated as an embedding; otherwise, as an image.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> np.ndarray:
        pass


if __name__ == "__main__":
    import asyncio

    micro_sam_trainer = MicroSamTrainer.func_or_class()

    async def test():
        response = await micro_sam_trainer.start_fit()
        print(response)

    asyncio.run(test())
