from dftracer.python.ai_common import *

ai = AI()
comm = ai.comm
compute = ai.compute
data = ai.data
dataloader = ai.dataloader
device = ai.device
checkpoint = ai.checkpoint
pipeline = ai.pipeline
checkpoint = ai.checkpoint

__all__ = [
    "INIT_NAME",
    "ITER_COUNT_NAME",
    "CommunicationEvent",
    "CheckpointEvent",
    "ComputeEvent",
    "DataEvent",
    "DataLoaderEvent",
    "DeviceEvent",
    "IOEvent",
    "PipelineEvent",
    "ProfileCategory",
    "ai",
    "checkpoint",
    "comm",
    "compute",
    "data",
    "dataloader",
    "device",
    "get_iter_block_name",
    "get_iter_handle_name",
    "pipeline",
]
