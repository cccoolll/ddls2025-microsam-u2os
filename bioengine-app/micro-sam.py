from ray import serve
from hypha_rpc.utils.schema import schema_method
import numpy as np
from pydantic import Field


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {
            "pip": ["torch"],
            "conda": ["micro-sam"]
        },
    },
)
class MicroSamTrainer:
    def __init__(self):
        pass

    async def _some_internal_function(self):
        pass

    @schema_method
    async def train(self, n_epochs: int = 1):
        import torch

        pass

    @schema_method
    async def encode_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @schema_method
    async def download_mask_decoder(self):
        pass

    @schema_method
    async def segment_all(self, image_or_embedding: np.ndarray, embedding: bool = False) -> np.ndarray:
        pass


if __name__ == "__main__":
    import asyncio

    micro_sam_trainer = MicroSamTrainer.func_or_class()

    async def test():
        await micro_sam_trainer.train()

    asyncio.run(test())
    