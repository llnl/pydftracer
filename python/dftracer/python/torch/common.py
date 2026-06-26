from typing import Any

from dftracer.python.common import TagDType, TagType, TagValue

dftracer = None  # type: ignore


# 1. Custom Profiler Plugin (handler)
def trace_handler(profiler_result: Any) -> None:
    global dftracer
    events = profiler_result.events()
    # Print attributes for each event
    dftracer.get_instance().enter_event()  # type: ignore
    for _i, event in enumerate(events):
        # Extract kernel name from event.key
        key = event.key
        # Check available attributes of time_range
        start_time_us = int(event.time_range.start)
        duration_us = int(event.time_range.elapsed_us())
        int_args = {}
        int_args["device"] = TagValue(
            event.device_type, TagDType.INT, TagType.KEY
        ).value()
        int_args["cpu_memory"] = TagValue(
            event.cpu_memory_usage, TagDType.INT, TagType.KEY
        ).value()
        int_args["is_remote"] = TagValue(
            event.is_remote, TagDType.INT, TagType.KEY
        ).value()
        int_args["device_memory_usage"] = TagValue(
            event.device_memory_usage, TagDType.INT, TagType.KEY
        ).value()
        int_args["input_size"] = TagValue(
            sum(len(s) for s in event.input_shapes) if event.input_shapes else 0,
            TagDType.INT,
            TagType.KEY,
        ).value()
        float_args = {}
        float_args["total_cpu_percent"] = TagValue(
            event.total_cpu_percent, TagDType.FLOAT, TagType.KEY
        ).value()
        float_args["total_device_percent"] = TagValue(
            event.total_device_percent, TagDType.FLOAT, TagType.KEY
        ).value()

        dftracer.get_instance().log_event(  # type: ignore
            name=key,
            cat="PP",
            start_time=start_time_us,
            duration=duration_us,
            int_args=int_args,
            float_args=float_args,
            string_args={},
        )
    dftracer.get_instance().exit_event()  # type: ignore
