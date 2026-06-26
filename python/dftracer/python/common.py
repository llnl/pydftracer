import inspect
import logging
import os
import signal
import sys
from collections.abc import Iterable, Sequence
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from dftracer.python.env import *
from typing_extensions import ParamSpec

P = ParamSpec("P")  # For capturing function parameters
R = TypeVar("R")  # For capturing function return types
T = TypeVar("T")  # For generic iterator types

# Type alias for TagValue return type
TagValueTuple = Tuple[int, Union[int, float, str]]


class ProfilerProtocol(Protocol):
    """Protocol defining the interface for profiler implementations."""

    def initialize(
        self,
        log_file: Optional[str] = None,
        data_dirs: Optional[str] = None,
        process_id: int = -1,
    ) -> None:
        """Initialize the profiler with configuration."""
        ...  # pragma: no cover

    def get_time(self) -> int:
        """Get current time for profiling."""
        ...  # pragma: no cover

    def enter_event(self) -> None:
        """Mark entry into an event."""
        ...  # pragma: no cover

    def exit_event(self) -> None:
        """Mark exit from an event."""
        ...  # pragma: no cover

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        int_args: Optional[Dict[str, TagValueTuple]] = None,
        string_args: Optional[Dict[str, TagValueTuple]] = None,
        float_args: Optional[Dict[str, TagValueTuple]] = None,
    ) -> None:
        """Log a profiling event."""
        ...  # pragma: no cover

    def log_metadata_event(self, key: str, value: str) -> None:
        """Log a metadata event."""
        ...  # pragma: no cover

    def finalize(self) -> None:
        """Finalize the profiler and release resources."""
        ...  # pragma: no cover


class NoOpProfiler:
    """Fallback no-operation profiler when pydftracer is not available."""

    def initialize(
        self,
        log_file: Optional[str] = None,
        data_dirs: Optional[str] = None,
        process_id: int = -1,
    ) -> None:
        pass

    def get_time(self) -> int:
        return 0

    def enter_event(self) -> None:
        pass

    def exit_event(self) -> None:
        pass

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        int_args: Optional[Dict[str, TagValueTuple]] = None,
        string_args: Optional[Dict[str, TagValueTuple]] = None,
        float_args: Optional[Dict[str, TagValueTuple]] = None,
    ) -> None:
        pass

    def log_metadata_event(self, key: str, value: str) -> None:
        pass

    def finalize(self) -> None:
        pass


# Profiler instance - will be set by logger.py during initialization
# Initialize with NoOpProfiler as default fallback
profiler: ProfilerProtocol = NoOpProfiler()


class TagType(Enum):
    KEY = 0
    VALUE = 1
    IGNORE = 2

    def __int__(self) -> int:
        return self.value


class TagDType(Enum):
    INT = 0
    FLOAT = 1
    STRING = 2

    def __int__(self) -> int:
        return self.value


class TagValue:
    def __init__(
        self,
        value: Any,
        dtype: TagDType = TagDType.STRING,
        tag_type: TagType = TagType.KEY,
    ) -> None:
        self._dtype = dtype
        self._value: Union[int, float, str]
        self._tag_type = tag_type
        if self._dtype == TagDType.INT:
            self._value = int(value)
        elif self._dtype == TagDType.FLOAT:
            self._value = float(value)
        else:
            self._value = str(value)

    def value(self) -> TagValueTuple:
        return (int(self._tag_type), self._value)


def capture_signal(signal_number: int, frame: Any) -> None:
    dftracer.get_instance().finalize()
    sys.exit(signal_number)


if DFTRACER_ENABLE:
    signal.signal(signal.SIGABRT, capture_signal)
    signal.signal(signal.SIGINT, capture_signal)
    signal.signal(signal.SIGTERM, capture_signal)


def setup_logger(
    name: str, log_file: str, formatter: str, level: int = logging.INFO
) -> logging.Logger:
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class dftracer:
    __instance: Optional["dftracer"] = None

    def __init__(self) -> None:
        self.logger: Optional[ProfilerProtocol] = None
        self.dbg_logging: Optional[logging.Logger] = None
        dftracer.__instance = self

    @classmethod
    def get_instance(cls) -> "dftracer":
        """Static access method."""
        if dftracer.__instance is None:
            dftracer()
        # for linter (e.g. mypy) to understand that this won't be None
        assert dftracer.__instance is not None
        return dftracer.__instance

    @staticmethod
    def initialize_log(
        logfile: Optional[str] = None,
        data_dir: Optional[str] = None,
        process_id: int = -1,
    ) -> "dftracer":
        instance = dftracer.get_instance()
        global profiler
        log_file_path = None
        if logfile:
            log_file_path = Path(logfile)
        outfile = "dft.log"
        if DFTRACER_ENABLE:
            if log_file_path:
                os.makedirs(log_file_path.parent, exist_ok=True)
                outfile = os.path.join(log_file_path.parent, "dft.log")
        log_level = logging.ERROR
        if DFTRACER_LOG_LEVEL == "DEBUG":
            log_level = logging.DEBUG
        elif DFTRACER_LOG_LEVEL == "INFO":
            log_level = logging.INFO
        elif DFTRACER_LOG_LEVEL == "WARN":
            log_level = logging.WARN
        instance.dbg_logging = setup_logger(
            name="dftracer_dbg",
            log_file=outfile,
            formatter="[DFTRACER_PY %(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
            level=log_level,
        )
        instance.dbg_logging.debug(
            f"logger.initialize_log {logfile} {data_dir} {process_id}"
        )
        if DFTRACER_ENABLE:
            instance.logger = profiler
            instance.dbg_logging.debug(
                f"logger.initialize {logfile} {data_dir} {process_id}"
            )
            instance.logger.initialize(
                log_file=logfile, data_dirs=data_dir, process_id=process_id
            )
        return instance

    def get_time(self) -> int:
        if DFTRACER_ENABLE and self.logger:
            t = self.logger.get_time()
            if self.dbg_logging:
                self.dbg_logging.debug(f"logger.get_time {t}")
            return t
        return 0

    def enter_event(self) -> None:
        if DFTRACER_ENABLE and self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug("logger.enter_event")
            return self.logger.enter_event()

    def exit_event(self) -> None:
        if DFTRACER_ENABLE and self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug("logger.exit_event")
            self.logger.exit_event()

    def log_event(
        self,
        name: str,
        cat: str,
        start_time: int,
        duration: int,
        int_args: Optional[Dict[str, TagValueTuple]] = None,
        float_args: Optional[Dict[str, TagValueTuple]] = None,
        string_args: Optional[Dict[str, TagValueTuple]] = None,
    ) -> None:
        if DFTRACER_ENABLE and self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug(
                    f"logger.log_event {name} {cat} {start_time} {duration} int={int_args} str={string_args} float={float_args}"
                )
            self.logger.log_event(
                name=name,
                cat=cat,
                start_time=start_time,
                duration=duration,
                int_args=int_args or {},
                float_args=float_args or {},
                string_args=string_args or {},
            )

    def log_metadata_event(self, key: str, value: str) -> None:
        if DFTRACER_ENABLE and self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug(f"logger.log_metadata_event {key} {value}")
            self.logger.log_metadata_event(key=key, value=value)

    def finalize(self) -> None:
        if DFTRACER_ENABLE and self.logger:
            if self.dbg_logging:
                self.dbg_logging.debug("logger.finalize")
            self.logger.finalize()


def get_default_args(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class dft_fn:
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
        self._enable = enable
        self._cat = cat
        self._arguments_int: Dict[str, TagValueTuple] = {}
        self._arguments_float: Dict[str, TagValueTuple] = {}
        self._arguments_string: Dict[str, TagValueTuple] = {}
        self._t1: int = 0
        self._t2: int = 0
        self._flush: bool = False

        if not name:
            name = inspect.stack()[1].function
        self._name = name

        if DFTRACER_ENABLE and self._enable:
            if epoch is not None:
                self._arguments_int["epoch"] = TagValue(
                    epoch, TagDType.INT, TagType.KEY
                ).value()
            if step is not None:
                self._arguments_int["step"] = TagValue(
                    step, TagDType.INT, TagType.KEY
                ).value()
            if image_idx is not None:
                self._arguments_int["image_idx"] = TagValue(
                    image_idx, TagDType.INT, TagType.KEY
                ).value()
            if image_size is not None:
                self._arguments_float["image_size"] = TagValue(
                    image_size, TagDType.FLOAT, TagType.KEY
                ).value()
            self.reset()

    def __enter__(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if DFTRACER_ENABLE and self._enable:
            if not self._flush:
                self.flush()

    def update(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        image_idx: Optional[int] = None,
        image_size: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            if epoch is not None:
                self._arguments_int["epoch"] = TagValue(
                    epoch, TagDType.INT, TagType.KEY
                ).value()
            if step is not None:
                self._arguments_int["step"] = TagValue(
                    step, TagDType.INT, TagType.KEY
                ).value()
            if image_idx is not None:
                self._arguments_int["image_idx"] = TagValue(
                    image_idx, TagDType.INT, TagType.KEY
                ).value()
            if image_size is not None:
                self._arguments_float["image_size"] = TagValue(
                    image_size, TagDType.FLOAT, TagType.KEY
                ).value()
            if args is not None:
                for key, value in args.items():
                    if isinstance(value, TagValue):
                        new_value = value._value
                    else:
                        new_value = value
                    if isinstance(new_value, int):
                        self._arguments_int[key] = TagValue(
                            new_value, TagDType.INT, TagType.KEY
                        ).value()
                    elif isinstance(new_value, float):
                        self._arguments_float[key] = TagValue(
                            new_value, TagDType.FLOAT, TagType.KEY
                        ).value()
                    else:
                        self._arguments_string[key] = TagValue(
                            str(new_value), TagDType.STRING, TagType.KEY
                        ).value()
        return self

    def flush(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t2 = dftracer.get_instance().get_time()
            dftracer.get_instance().log_event(
                name=self._name,
                cat=self._cat,
                start_time=self._t1,
                duration=self._t2 - self._t1,
                int_args=self._arguments_int,
                float_args=self._arguments_float,
                string_args=self._arguments_string,
            )
            dftracer.get_instance().exit_event()
            self._flush = True
        return self

    def reset(self) -> "dft_fn":
        if DFTRACER_ENABLE and self._enable:
            self._t1 = dftracer.get_instance().get_time()
            dftracer.get_instance().enter_event()
            self._t2 = self._t1
            self._flush = False
        return self

    @overload
    def log(
        self, func: Callable[P, R], name: Optional[str] = None
    ) -> Callable[P, R]: ...

    @overload
    def log(
        self, func: None = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def log(
        self, func: Optional[Callable[P, R]] = None, name: Optional[str] = None
    ) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(f: Callable[P, R]) -> Callable[P, R]:
            _name = name if name else f.__qualname__
            arg_names: List[str] = []
            if DFTRACER_ENABLE and self._enable:
                arg_names = inspect.getfullargspec(f)[0]
                self._arguments_int = {}
                self._arguments_float = {}
                self._arguments_string = {}

            @wraps(f)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if DFTRACER_ENABLE and self._enable:
                    if len(arg_names) > 0:
                        if "self" == arg_names[0]:
                            if hasattr(args[0], "epoch"):
                                self._arguments_int["epoch"] = TagValue(
                                    args[0].epoch, TagDType.INT, TagType.KEY
                                ).value()
                            if hasattr(args[0], "step"):
                                self._arguments_int["step"] = TagValue(
                                    args[0].step, TagDType.INT, TagType.KEY
                                ).value()
                            if hasattr(args[0], "image_size"):
                                self._arguments_float["image_size"] = TagValue(
                                    args[0].image_size, TagDType.FLOAT, TagType.KEY
                                ).value()
                            if hasattr(args[0], "image_idx"):
                                self._arguments_int["image_idx"] = TagValue(
                                    args[0].image_idx, TagDType.INT, TagType.KEY
                                ).value()
                        full_args = dict(zip(arg_names[1:], args[1:]))
                        full_args.update(kwargs)
                        full_args.update(get_default_args(f))

                        for param_name, value in full_args.items():
                            if param_name == "epoch":
                                self._arguments_int["epoch"] = TagValue(
                                    value, TagDType.INT, TagType.KEY
                                ).value()
                            elif param_name == "image_idx":
                                self._arguments_int["image_idx"] = TagValue(
                                    value, TagDType.INT, TagType.KEY
                                ).value()
                            elif param_name == "image_size":
                                self._arguments_float["image_size"] = TagValue(
                                    value, TagDType.FLOAT, TagType.KEY
                                ).value()
                            elif param_name == "step":
                                self._arguments_int["step"] = TagValue(
                                    value, TagDType.INT, TagType.KEY
                                ).value()

                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()
                x = f(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        int_args=self._arguments_int,
                        float_args=self._arguments_float,
                        string_args=self._arguments_string,
                    )
                    dftracer.get_instance().exit_event()
                return x

            return wrapper

        return _decorator(func) if callable(func) else _decorator

    def log_metadata(self, key: str, value: str) -> None:
        if DFTRACER_ENABLE and self._enable:
            dftracer.get_instance().log_metadata_event(key=key, value=value)

    def iter(
        self,
        func: Union[Sequence[T], Iterable[T], Generator[T, Any, Any]],
        name: str = "loop",
        iter_name: str = "step",
    ) -> Generator[T, Any, Any]:
        iter_val = 1
        _name = ""
        kernel_name = ""
        start = 0
        if DFTRACER_ENABLE and self._enable:
            _name = f"{name}.iter"
            kernel_name = f"{name}.yield"
            dftracer.get_instance().enter_event()
            start = dftracer.get_instance().get_time()
            self._arguments_int = {}
            self._arguments_float = {}
            self._arguments_string = {}

        for v in func:
            if DFTRACER_ENABLE and self._enable:
                end = dftracer.get_instance().get_time()
                self._arguments_int[iter_name] = TagValue(
                    iter_val, TagDType.INT, TagType.KEY
                ).value()
                dftracer.get_instance().log_event(
                    name=_name,
                    cat=self._cat,
                    start_time=start,
                    duration=end - start,
                    int_args=self._arguments_int,
                    float_args=self._arguments_float,
                    string_args=self._arguments_string,
                )
                dftracer.get_instance().exit_event()
                dftracer.get_instance().enter_event()
                t0 = dftracer.get_instance().get_time()
            yield v
            if DFTRACER_ENABLE and self._enable:
                t1 = dftracer.get_instance().get_time()
                dftracer.get_instance().log_event(
                    name=kernel_name,
                    cat=self._cat,
                    start_time=t0,
                    duration=t1 - t0,
                    int_args=self._arguments_int,
                    float_args=self._arguments_float,
                    string_args=self._arguments_string,
                )
                dftracer.get_instance().exit_event()

                iter_val += 1
                dftracer.get_instance().enter_event()
                start = dftracer.get_instance().get_time()

        if DFTRACER_ENABLE and self._enable:
            dftracer.get_instance().exit_event()

    @overload
    def log_init(
        self, init: Callable[P, R], name: Optional[str] = None
    ) -> Callable[P, R]: ...

    @overload
    def log_init(
        self, init: None = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def log_init(
        self, init: Optional[Callable[P, R]] = None, name: Optional[str] = None
    ) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(fn: Callable[P, R]) -> Callable[P, R]:
            _name = name if name else fn.__qualname__
            arg_names: List[str] = []
            if DFTRACER_ENABLE and self._enable:
                arg_names = inspect.getfullargspec(fn)[0]
                self._arguments_int = {}
                self._arguments_float = {}
                self._arguments_string = {}

            @wraps(fn)
            def new_init(*args: Any, **kwargs: Any) -> Any:
                if DFTRACER_ENABLE and self._enable:
                    arg_values = dict(zip(arg_names, args))
                    arg_values.update(kwargs)
                    arg_values.update(get_default_args(fn))
                    if "epoch" in arg_values and arg_values["epoch"] is not None:
                        self._arguments_int["epoch"] = TagValue(
                            arg_values["epoch"], TagDType.INT, TagType.KEY
                        ).value()
                    if (
                        "image_idx" in arg_values
                        and arg_values["image_idx"] is not None
                    ):
                        self._arguments_int["image_idx"] = TagValue(
                            arg_values["image_idx"], TagDType.INT, TagType.KEY
                        ).value()
                    if (
                        "image_size" in arg_values
                        and arg_values["image_size"] is not None
                    ):
                        self._arguments_float["image_size"] = TagValue(
                            arg_values["image_size"], TagDType.FLOAT, TagType.KEY
                        ).value()
                    if "step" in arg_values and arg_values["step"] is not None:
                        self._arguments_int["step"] = TagValue(
                            arg_values["step"], TagDType.INT, TagType.KEY
                        ).value()
                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()
                fn(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        int_args=self._arguments_int,
                        float_args=self._arguments_float,
                        string_args=self._arguments_string,
                    )
                    dftracer.get_instance().exit_event()

            return new_init

        return _decorator(init) if callable(init) else _decorator

    @overload
    def log_static(
        self, func: Callable[P, R], name: Optional[str] = None
    ) -> Callable[P, R]: ...

    @overload
    def log_static(
        self, func: None = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def log_static(
        self, func: Optional[Callable[P, R]] = None, name: Optional[str] = None
    ) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
        # CC BY-SA 4.0 https://stackoverflow.com/a/60832711
        def _decorator(f: Callable[P, R]) -> Callable[P, R]:
            # Extract actual function if we received a staticmethod object
            if isinstance(f, staticmethod):
                actual_func = cast(Callable[P, R], f.__func__)
            else:
                actual_func = f

            _name = name if name else actual_func.__qualname__

            @wraps(actual_func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if DFTRACER_ENABLE and self._enable:
                    start = dftracer.get_instance().get_time()
                    dftracer.get_instance().enter_event()
                x = actual_func(*args, **kwargs)
                if DFTRACER_ENABLE and self._enable:
                    end = dftracer.get_instance().get_time()
                    dftracer.get_instance().log_event(
                        name=_name,
                        cat=self._cat,
                        start_time=start,
                        duration=end - start,
                        int_args=self._arguments_int,
                        float_args=self._arguments_float,
                        string_args=self._arguments_string,
                    )
                    dftracer.get_instance().exit_event()
                return x

            return wrapper

        return _decorator(func) if callable(func) else _decorator
