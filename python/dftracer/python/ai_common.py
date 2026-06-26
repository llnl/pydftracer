import functools
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, overload

from dftracer.python.common import *
from dftracer.python.env import *

dft_fn = None  # type: ignore
dftracer = None  # type: ignore

if sys.version_info >= (3, 11):
    from enum import StrEnum as StringEnum
    from enum import auto
    # StrEnum in 3.11 already lower-case
else:
    from enum import Enum, auto

    # MIT License: https://github.com/irgeek/StrEnum/blob/master/strenum/__init__.py
    class StringEnum(str, Enum):
        # this is a specific version that will lowercase the values

        def __new__(cls, value: Any, *args: Any, **kwargs: Any) -> "StringEnum":
            if not isinstance(value, (str, auto)):
                raise TypeError(  # pragma: no cover
                    f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
                )
            return super().__new__(cls, value, *args, **kwargs)

        def __str__(self) -> str:
            return str(self.value)

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: List[Any]
        ) -> str:
            return name.lower()


ITER_COUNT_NAME = "count"
INIT_NAME = "init"
BLOCK_NAME = "block"
ITER_NAME = "iter"
CTX_SEPARATOR = "."
ROOT_NAME = "ai_root"
ROOT_CAT = "ai_root"
START_METADATA_NAME = "start"
STOP_METADATA_NAME = "end"


def get_iter_block_name(name: str) -> str:
    return (
        f"{name}{CTX_SEPARATOR}{BLOCK_NAME}"
        if not name.endswith(f"{CTX_SEPARATOR}{BLOCK_NAME}")
        else name
    )


def get_iter_handle_name(name: str) -> str:
    return (
        f"{name}{CTX_SEPARATOR}{ITER_NAME}"
        if not name.endswith(f"{CTX_SEPARATOR}{ITER_NAME}")
        else name
    )


class _DFTracerAI:
    def __init__(
        self,
        cat: str,
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ) -> None:
        if not name:
            name = cat
        global dft_fn
        self.profiler = dft_fn(
            cat=cat,
            name=name,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )

    @overload
    def __call__(
        self,
        fn: Callable[P, R],
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Callable[P, R]: ...

    @overload
    def __call__(
        self,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> "DFTracerAI": ...

    def __call__(
        self,
        fn: Optional[Callable[P, R]] = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Union[Callable[P, R], "DFTracerAI"]:
        if epoch is not None:
            self.profiler._arguments_int["epoch"] = TagValue(
                epoch, TagDType.INT, TagType.KEY
            ).value()
        if step is not None:
            self.profiler._arguments_int["step"] = TagValue(
                step, TagDType.INT, TagType.KEY
            ).value()
        if image_idx is not None:
            self.profiler._arguments_int["image_idx"] = TagValue(
                image_idx, TagDType.INT, TagType.KEY
            ).value()
        if image_size is not None:
            self.profiler._arguments_float["image_size"] = TagValue(
                image_size, TagDType.FLOAT, TagType.KEY
            ).value()
        args = args or {}
        for key, value in args.items():
            if isinstance(value, TagValue):
                new_value = value._value
            else:
                new_value = value
            if isinstance(new_value, int):
                self.profiler._arguments_int[key] = TagValue(
                    new_value, TagDType.INT, TagType.KEY
                ).value()
            elif isinstance(new_value, float):
                self.profiler._arguments_float[key] = TagValue(
                    new_value, TagDType.FLOAT, TagType.KEY
                ).value()
            else:
                self.profiler._arguments_string[key] = TagValue(
                    str(new_value), TagDType.STRING, TagType.KEY
                ).value()

        is_enabled = self.profiler._enable if enable is None else enable

        if fn:

            def _decorator(f: Callable[P, R]) -> Callable[P, R]:
                @functools.wraps(f)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    if is_enabled:
                        with self:
                            return f(*args, **kwargs)
                    return f(*args, **kwargs)

                return wrapper

            return _decorator(fn)
        else:
            return DFTracerAI(
                cat=self.profiler._cat,
                name=self.profiler._name,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                enable=is_enabled,
            )

    def __enter__(self) -> "_DFTracerAI":
        self.profiler.__enter__()
        # Reset flush state to ensure proper event logging on each context manager entry.
        # The underlying DFTracer logger was designed for one-time use objects, but this
        # class acts as a singleton. Without this reset, events won't flush after the
        # first __exit__ call due to DFTracer's internal _flush state management.
        self.profiler._flush = False
        return self

    def get_time(self) -> int:
        global dftracer
        return dftracer.get_instance().get_time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def start(self, metadata: bool = False) -> None:
        global dftracer
        if metadata:
            time = dftracer.get_instance().get_time()
            self.profiler.log_metadata(
                key=f"{self.profiler._name}{CTX_SEPARATOR}{START_METADATA_NAME}",
                value=str(time),
            )
        else:
            self.__enter__()

    def stop(self, metadata: bool = False) -> None:
        global dftracer
        if metadata:
            time = dftracer.get_instance().get_time()
            self.profiler.log_metadata(
                key=f"{self.profiler._name}{CTX_SEPARATOR}{STOP_METADATA_NAME}",
                value=str(time),
            )
        else:
            self.__exit__(None, None, None)

    def reset(self) -> None:
        self.profiler.reset()

    def enable(self) -> None:
        self.profiler._enable = True

    def disable(self) -> None:
        self.profiler._enable = False

    @property
    def cat(self) -> str:
        return self.profiler._cat

    @property
    def name(self) -> str:
        return self.profiler._name

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> "_DFTracerAI":
        if args is None:
            args = {}
        if DFTRACER_ENABLE and self.profiler._enable:
            if epoch is not None:
                self.profiler._arguments_int["epoch"] = TagValue(
                    epoch, TagDType.INT, TagType.KEY
                ).value()
            if step is not None:
                self.profiler._arguments_int["step"] = TagValue(
                    step, TagDType.INT, TagType.KEY
                ).value()
            if image_idx is not None:
                self.profiler._arguments_int["image_idx"] = TagValue(
                    image_idx, TagDType.INT, TagType.KEY
                ).value()
            if image_size is not None:
                self.profiler._arguments_float["image_size"] = TagValue(
                    image_size, TagDType.FLOAT, TagType.KEY
                ).value()
            for key, value in args.items():
                if isinstance(value, TagValue):
                    new_value = value._value
                else:
                    new_value = value
                if isinstance(new_value, int):
                    self.profiler._arguments_int[key] = TagValue(
                        new_value, TagDType.INT, TagType.KEY
                    ).value()
                elif isinstance(new_value, float):
                    self.profiler._arguments_float[key] = TagValue(
                        new_value, TagDType.FLOAT, TagType.KEY
                    ).value()
                else:
                    self.profiler._arguments_string[key] = TagValue(
                        str(new_value), TagDType.STRING, TagType.KEY
                    ).value()
        return self

    def iter(
        self,
        iterator: Iterator,
        *,
        include_block: bool = True,
        include_iter: bool = True,
        iter_name: Optional[str] = None,
        block_name: Optional[str] = None,
    ) -> Iterator:
        global dftracer
        iter_name = iter_name or get_iter_handle_name(self.profiler._name)
        block_name = block_name or get_iter_block_name(self.profiler._name)
        iter_val = 1

        start: int = 0
        if DFTRACER_ENABLE and self.profiler._enable:
            dftracer.get_instance().enter_event()
            start = dftracer.get_instance().get_time()

        for v in iterator:
            if DFTRACER_ENABLE and self.profiler._enable:
                end = dftracer.get_instance().get_time()
                self.profiler._arguments_int[ITER_COUNT_NAME] = TagValue(
                    iter_val, TagDType.INT, TagType.KEY
                ).value()
                if include_iter:
                    dftracer.get_instance().log_event(
                        name=iter_name,
                        cat=self.profiler._cat,
                        start_time=start,
                        duration=end - start,
                        int_args=self.profiler._arguments_int,
                        float_args=self.profiler._arguments_float,
                        string_args=self.profiler._arguments_string,
                    )
                dftracer.get_instance().exit_event()
                dftracer.get_instance().enter_event()
                t0 = dftracer.get_instance().get_time()

            yield v

            if DFTRACER_ENABLE and self.profiler._enable:
                t1 = dftracer.get_instance().get_time()
                if include_block:
                    dftracer.get_instance().log_event(
                        name=block_name,
                        cat=self.profiler._cat,
                        start_time=t0,
                        duration=t1 - t0,
                        int_args=self.profiler._arguments_int,
                        float_args=self.profiler._arguments_float,
                        string_args=self.profiler._arguments_string,
                    )
                dftracer.get_instance().exit_event()

                iter_val += 1
                dftracer.get_instance().enter_event()
                start = dftracer.get_instance().get_time()
        if DFTRACER_ENABLE and self.profiler._enable:
            dftracer.get_instance().exit_event()


class DFTracerAI(_DFTracerAI):
    def __init__(
        self,
        *,
        cat: str,
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=cat,
            name=name,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self._children: Dict[str, DFTracerAI] = {}

    def create_children(self, names: Dict[str, str]) -> None:
        for attr, name in names.items():
            epoch_tuple = self.profiler._arguments_int.get("epoch")
            step_tuple = self.profiler._arguments_int.get("step")
            image_idx_tuple = self.profiler._arguments_int.get("image_idx")
            image_size_tuple = self.profiler._arguments_float.get("image_size")

            tracer = DFTracerAI(
                cat=self.profiler._cat,
                name=name,
                epoch=int(epoch_tuple[1]) if epoch_tuple else None,
                step=int(step_tuple[1]) if step_tuple else None,
                image_idx=int(image_idx_tuple[1]) if image_idx_tuple else None,
                image_size=float(image_size_tuple[1]) if image_size_tuple else None,
                enable=self.profiler._enable,
            )
            setattr(self, attr, tracer)
            self._children[attr] = tracer

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> "DFTracerAI":
        super().update(
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            args=args,
        )
        for tracer in self._children.values():
            tracer.update(
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                args=args,
            )
        return self

    def enable(self) -> None:
        super().enable()
        for tracer in self._children.values():
            tracer.enable()

    def disable(self) -> None:
        super().disable()
        for tracer in self._children.values():
            tracer.disable()

    def derive(self, name: str) -> "DFTracerAI":
        _name = f"{self.profiler._name}{CTX_SEPARATOR}{name}"
        if _name in self._children:
            return self._children[_name]

        epoch_tuple = self.profiler._arguments_int.get("epoch")
        step_tuple = self.profiler._arguments_int.get("step")
        image_idx_tuple = self.profiler._arguments_int.get("image_idx")
        image_size_tuple = self.profiler._arguments_float.get("image_size")

        child = DFTracerAI(
            cat=self.profiler._cat,
            name=_name,
            epoch=int(epoch_tuple[1]) if epoch_tuple else None,
            step=int(step_tuple[1]) if step_tuple else None,
            image_idx=int(image_idx_tuple[1]) if image_idx_tuple else None,
            image_size=float(image_size_tuple[1]) if image_size_tuple else None,
            enable=self.profiler._enable,
        )
        self._children[_name] = child
        return child

    @overload
    def init(
        self,
        fn: Callable[P, R],
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Callable[P, R]: ...

    @overload
    def init(
        self,
        fn: None = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> "DFTracerAI": ...

    def init(
        self,
        fn: Optional[Callable[P, R]] = None,
        *,
        enable: Optional[bool] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Union[Callable[P, R], "DFTracerAI"]:
        _name = f"{self.profiler._name}{CTX_SEPARATOR}{INIT_NAME}"
        if _name in self._children:
            return self._children[_name](  # type: ignore[misc]
                fn=fn,  # type: ignore[arg-type]
                enable=enable,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                image_size=image_size,
                args=args,
            )
        epoch_tuple = self.profiler._arguments_int.get("epoch")
        step_tuple = self.profiler._arguments_int.get("step")
        image_idx_tuple = self.profiler._arguments_int.get("image_idx")
        image_size_tuple = self.profiler._arguments_float.get("image_size")

        child = DFTracerAI(
            cat=self.profiler._cat,
            name=_name,
            epoch=int(epoch_tuple[1]) if epoch_tuple else None,
            step=int(step_tuple[1]) if step_tuple else None,
            image_idx=int(image_idx_tuple[1]) if image_idx_tuple else None,
            image_size=float(image_size_tuple[1]) if image_size_tuple else None,
            enable=self.profiler._enable,
        )
        self._children[_name] = child
        return child(  # type: ignore[misc]
            fn=fn,  # type: ignore[arg-type]
            enable=enable,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            args=args,
        )


# Enumerations


class ProfileCategory(StringEnum):
    COMPUTE = auto()
    DATA = auto()
    DATALOADER = auto()
    COMM = auto()
    DEVICE = auto()
    CHECKPOINT = auto()
    PIPELINE = auto()


class ComputeEvent(StringEnum):
    FORWARD = auto()
    BACKWARD = auto()
    STEP = auto()


class DataEvent(StringEnum):
    PREPROCESS = auto()
    ITEM = auto()


class DataLoaderEvent(StringEnum):
    FETCH = auto()


# Terminologies are taken from https://docs.pytorch.org/docs/stable/distributed.html
class CommunicationEvent(StringEnum):
    SEND = auto()
    RECEIVE = auto()
    BARRIER = auto()
    BCAST = auto()
    REDUCE = auto()
    ALL_REDUCE = auto()
    GATHER = auto()
    ALL_GATHER = auto()
    SCATTER = auto()
    REDUCE_SCATTER = auto()
    ALL_TO_ALL = auto()


class DeviceEvent(StringEnum):
    TRANSFER = auto()


class PipelineEvent(StringEnum):
    EPOCH = auto()
    TRAIN = auto()
    EVALUATE = auto()
    TEST = auto()


class CheckpointEvent(StringEnum):
    CAPTURE = auto()
    RESTART = auto()


class IOEvent(StringEnum):
    OPEN = auto()
    READ = auto()
    WRITE = auto()
    CLOSE = auto()


class IO(DFTracerAI):
    """Tracer for I/O operations on a data object (open, read, write, close).

    Accessible as ``data.io`` or ``checkpoint.io``. Each operation maps to a
    dedicated child tracer so that load/open, read, write, and delete/close
    phases are captured as separate, named events in the trace.

    Events:
        open  -- opening or loading a data object (e.g. file open, dataset load)
        read  -- reading data from an already-open object
        write -- writing data to an object
        close -- closing or deleting a data object
    """

    open: DFTracerAI
    read: DFTracerAI
    write: DFTracerAI
    close: DFTracerAI

    def __init__(
        self,
        *,
        cat: str,
        parent_name: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        io_name = f"{parent_name}{CTX_SEPARATOR}io"
        super().__init__(
            cat=cat,
            name=io_name,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "open": f"{io_name}{CTX_SEPARATOR}{IOEvent.OPEN}",
                "read": f"{io_name}{CTX_SEPARATOR}{IOEvent.READ}",
                "write": f"{io_name}{CTX_SEPARATOR}{IOEvent.WRITE}",
                "close": f"{io_name}{CTX_SEPARATOR}{IOEvent.CLOSE}",
            }
        )


class Compute(DFTracerAI):
    forward: DFTracerAI
    backward: DFTracerAI
    step: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.COMPUTE,
            name=ProfileCategory.COMPUTE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )

        self.create_children(
            {
                "forward": ComputeEvent.FORWARD,
                "backward": ComputeEvent.BACKWARD,
                "step": ComputeEvent.STEP,
            }
        )


class Data(DFTracerAI):
    preprocess: DFTracerAI
    item: DFTracerAI
    io: IO

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DATA,
            name=ProfileCategory.DATA,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "preprocess": DataEvent.PREPROCESS,
                "item": DataEvent.ITEM,
            }
        )
        self.io = IO(
            cat=ProfileCategory.DATA,
            parent_name=str(ProfileCategory.DATA),
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self._children["io"] = self.io


class DataLoader(DFTracerAI):
    fetch: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DATALOADER,
            name=ProfileCategory.DATALOADER,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "fetch": DataLoaderEvent.FETCH,
            }
        )


class Communication(DFTracerAI):
    send: DFTracerAI
    receive: DFTracerAI
    barrier: DFTracerAI
    bcast: DFTracerAI
    reduce: DFTracerAI
    all_reduce: DFTracerAI
    gather: DFTracerAI
    all_gather: DFTracerAI
    scatter: DFTracerAI
    reduce_scatter: DFTracerAI
    all_to_all: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.COMM,
            name=ProfileCategory.COMM,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "send": CommunicationEvent.SEND,
                "receive": CommunicationEvent.RECEIVE,
                "barrier": CommunicationEvent.BARRIER,
                "bcast": CommunicationEvent.BCAST,
                "reduce": CommunicationEvent.REDUCE,
                "all_reduce": CommunicationEvent.ALL_REDUCE,
                "gather": CommunicationEvent.GATHER,
                "all_gather": CommunicationEvent.ALL_GATHER,
                "scatter": CommunicationEvent.SCATTER,
                "reduce_scatter": CommunicationEvent.REDUCE_SCATTER,
                "all_to_all": CommunicationEvent.ALL_TO_ALL,
            }
        )


class Device(DFTracerAI):
    transfer: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.DEVICE,
            name=ProfileCategory.DEVICE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "transfer": DeviceEvent.TRANSFER,
            }
        )


class Checkpoint(DFTracerAI):
    capture: DFTracerAI
    restart: DFTracerAI
    io: IO

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.CHECKPOINT,
            name=ProfileCategory.CHECKPOINT,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "capture": CheckpointEvent.CAPTURE,
                "restart": CheckpointEvent.RESTART,
            }
        )
        self.io = IO(
            cat=ProfileCategory.CHECKPOINT,
            parent_name=str(ProfileCategory.CHECKPOINT),
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self._children["io"] = self.io


class Pipeline(DFTracerAI):
    epoch: DFTracerAI
    train: DFTracerAI
    evaluate: DFTracerAI
    test: DFTracerAI

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(
            cat=ProfileCategory.PIPELINE,
            name=ProfileCategory.PIPELINE,
            epoch=epoch,
            step=step,
            image_idx=image_idx,
            image_size=image_size,
            enable=enable,
        )
        self.create_children(
            {
                "epoch": PipelineEvent.EPOCH,
                "train": PipelineEvent.TRAIN,
                "evaluate": PipelineEvent.EVALUATE,
                "test": PipelineEvent.TEST,
            }
        )


# fmt: off
class AI(DFTracerAI):
    compute: Compute
    data: Data
    dataloader: DataLoader
    comm: Communication
    device: Device
    checkpoint: Checkpoint
    pipeline: Pipeline

    def __init__(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        enable: bool = True,
    ):
        super().__init__(cat=ROOT_CAT, name=ROOT_NAME, epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.compute = Compute(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.data = Data(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.dataloader = DataLoader(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.comm = Communication(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.device = Device(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.checkpoint = Checkpoint(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)
        self.pipeline = Pipeline(epoch=epoch, step=step, image_idx=image_idx, image_size=image_size, enable=enable)

        self._children = {
            "compute": self.compute,
            "data": self.data,
            "dataloader": self.dataloader,
            "comm": self.comm,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "pipeline": self.pipeline,
        }
# fmt: on
