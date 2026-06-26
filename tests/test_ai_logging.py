#!/usr/bin/env python
"""
AI Logging Test Suite for dftracer
"""

import os
import shutil
from dataclasses import dataclass
from time import sleep
from typing import Optional

import numpy as np
import pytest
from dftracer.python import ai, dftracer

from .utils import run_test_in_spawn_process, validate_log_files


@dataclass
class Args:
    log_dir: str
    data_dir: str
    num_files: int
    niter: int
    disable_ai_cat: Optional[str] = None
    epoch_as_metadata: bool = False
    record_size: int = 1048576


class IOHandler:
    @ai.data.item
    def read(self, filename: str):
        return np.load(filename)

    def write(self, filename: str, a):
        with open(filename, "wb") as f:
            np.save(f, a)


def data_gen(args: Args, io: IOHandler, data: np.ndarray):
    for i in range(args.num_files):
        io.write(f"{args.data_dir}/npz/{i}-of-{args.num_files}.npy", data)


@ai.dataloader.fetch
def read_data(args: Args, io: IOHandler, epoch: int):
    for i in range(args.num_files):
        yield io.read(f"{args.data_dir}/npz/{i}-of-{args.num_files}.npy")


@ai.data.preprocess.derive(name="collate")
def collate(data):
    return data


@ai.device.transfer
def transfer(data):
    sleep(0.1)
    return data


@ai.compute.forward
def forward(data):
    sleep(0.1)
    return 0.0


@ai.compute.backward
def backward():
    sleep(0.1)
    with ai.comm.all_reduce(enable=False):
        sleep(0.1)


@ai.compute
def compute(data):
    _ = forward(data)
    backward()
    return _


class Checkpoint:
    @ai.checkpoint.init
    def __init__(self):
        sleep(0.1)

    @ai.checkpoint.capture
    def capture(self, _):
        sleep(0.1)
        return _

    @ai.checkpoint.restart
    def restart(self, _):
        sleep(0.1)
        return _


class Hook:
    def before_step(self, *args, **kwargs):
        ai.compute.step.start()

    def after_step(self, *args, **kwargs):
        ai.compute.step.stop()


@ai.pipeline.train
def train(args: Args, hook: Hook):
    io = IOHandler()

    os.makedirs(f"{args.log_dir}/npz", exist_ok=True)
    os.makedirs(f"{args.data_dir}/npz", exist_ok=True)
    data = np.ones((args.record_size, 1), dtype=np.uint8)
    data_gen(args, io, data)

    checkpoint = Checkpoint()

    checkpoint.restart({})

    if args.epoch_as_metadata:
        for epoch in range(args.niter):
            ai.pipeline.epoch.start(metadata=True)
            for step, data in ai.dataloader.fetch.iter(
                enumerate(read_data(args, io, epoch))
            ):
                hook.before_step()
                data = collate(data)
                _ = transfer(data)
                _ = compute(data)
                hook.after_step()
                ai.update(step=step, epoch=epoch)
            ai.pipeline.epoch.stop(metadata=True)
    else:
        for epoch in ai.pipeline.epoch.iter(range(args.niter)):
            for step, data in ai.dataloader.fetch.iter(
                enumerate(read_data(args, io, epoch))
            ):
                hook.before_step()
                data = collate(data)
                _ = transfer(data)
                _ = compute(data)
                hook.after_step()
                ai.update(step=step, epoch=epoch)

    checkpoint.capture({})


def run_ai_logging_test(test_config):
    base_dir = os.path.join(os.path.dirname(__file__), "test_ai_logging_output")
    test_name = f"{test_config['name']}_niter{test_config['niter']}_files{test_config['num_files']}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_config['name']}.pfw")

    if test_config.get("disable_ai_cat") == "all":
        ai.disable()
    elif test_config.get("disable_ai_cat") == "dataloader":
        ai.dataloader.disable()
    elif test_config.get("disable_ai_cat") == "device":
        ai.device.disable()
    elif test_config.get("disable_ai_cat") == "compute":
        ai.compute.disable()
    elif test_config.get("disable_ai_cat") == "comm":
        ai.comm.disable()
    elif test_config.get("disable_ai_cat") == "ckpt":
        ai.checkpoint.disable()

    args = Args(
        log_dir=log_dir,
        data_dir=data_dir,
        disable_ai_cat=test_config.get("disable_ai_cat"),
        num_files=int(test_config["num_files"]),
        niter=int(test_config["niter"]),
        epoch_as_metadata=test_config.get("epoch_as_metadata", False),
        record_size=test_config.get("record_size", 1048576),
    )

    print(
        f"Running AI logging test {test_config['name']} with log file: {log_file}, args = {args}"
    )

    try:
        hook = Hook()
        df_logger = dftracer.initialize_log(
            logfile=log_file, data_dir=data_dir, process_id=-1
        )
        train(args, hook)
        df_logger.finalize()

        # Validate log files using the common utility
        expected_count = test_config.get("expected_events", 0)
        validate_log_files(log_file, test_config["name"], expected_count)
    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestAILogging:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "disabled",
                "num_files": 2,
                "niter": 1,
                "expected_events": 0,
                "env": {
                    "DFTRACER_ENABLE": "0",
                },
            },
            {
                "name": "normal",
                "num_files": 2,
                "niter": 3,
                "expected_events": 76,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "epoch_as_metadata",
                "num_files": 2,
                "niter": 3,
                "epoch_as_metadata": True,
                "expected_events": 76,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_all",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "all",
                "expected_events": 9,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_dataloader",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "dataloader",
                "expected_events": 61,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_device",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "device",
                "expected_events": 70,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_compute",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "compute",
                "expected_events": 52,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_ckpt",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "ckpt",
                "expected_events": 73,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
        ],
    )
    def test_ai_logging(self, test_config):
        run_test_in_spawn_process(run_ai_logging_test, test_config)


class DataIOHandler:
    """Exercises ai.data.io.{open,read,write,close} as decorators and context managers."""

    @ai.data.io.open
    def open(self, path: str):
        return path

    @ai.data.io.read
    def read(self, path: str):
        return np.ones((1024, 1), dtype=np.uint8)

    @ai.data.io.write
    def write(self, path: str, data: np.ndarray):
        with open(path, "wb") as f:
            np.save(f, data)

    @ai.data.io.close
    def close(self, path: str):
        pass


class CheckpointIOHandler:
    """Exercises ai.checkpoint.io.{open,read,write,close} as decorators and context managers."""

    @ai.checkpoint.io.open
    def open(self, path: str):
        return path

    @ai.checkpoint.io.read
    def load(self, path: str):
        return {}

    @ai.checkpoint.io.write
    def save(self, path: str, state: dict):
        pass

    @ai.checkpoint.io.close
    def delete(self, path: str):
        pass


def run_io_test(test_config):
    base_dir = os.path.join(os.path.dirname(__file__), "test_io_logging_output")
    test_name = test_config["name"]
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_name}.pfw")

    if test_config.get("disable_io_cat") == "data":
        ai.data.io.disable()
    elif test_config.get("disable_io_cat") == "checkpoint":
        ai.checkpoint.io.disable()

    sample_path = os.path.join(data_dir, "sample.npy")
    ckpt_path = os.path.join(data_dir, "model.pt")

    print(f"Running IO test {test_name} with log file: {log_file}")

    try:
        df_logger = dftracer.initialize_log(
            logfile=log_file, data_dir=data_dir, process_id=-1
        )

        # Exercise data IO operations via decorators
        data_io = DataIOHandler()
        data_io.open(sample_path)
        data = data_io.read(sample_path)
        data_io.write(sample_path, data)
        data_io.close(sample_path)

        # Exercise data IO operations via context managers
        with ai.data.io.open:
            pass
        with ai.data.io.read:
            pass
        with ai.data.io.write:
            pass
        with ai.data.io.close:
            pass

        # Exercise checkpoint IO operations via decorators
        ckpt_io = CheckpointIOHandler()
        ckpt_io.open(ckpt_path)
        ckpt_io.load(ckpt_path)
        ckpt_io.save(ckpt_path, {})
        ckpt_io.delete(ckpt_path)

        # Exercise checkpoint IO operations via context managers
        with ai.checkpoint.io.open:
            pass
        with ai.checkpoint.io.read:
            pass
        with ai.checkpoint.io.write:
            pass
        with ai.checkpoint.io.close:
            pass

        df_logger.finalize()

        expected_count = test_config.get("expected_events", 0)
        mode = test_config.get("mode", "exact")
        validate_log_files(log_file, test_name, expected_count, mode=mode)
    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestIOLogging:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "io_disabled",
                "expected_events": 0,
                "env": {
                    "DFTRACER_ENABLE": "0",
                },
            },
            {
                "name": "io_normal",
                "expected_events": 0,
                "mode": "exact",
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "io_disable_data_io",
                "disable_io_cat": "data",
                "expected_events": 0,
                "mode": "exact",
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "io_disable_checkpoint_io",
                "disable_io_cat": "checkpoint",
                "expected_events": 0,
                "mode": "exact",
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
        ],
    )
    def test_io_logging(self, test_config):
        run_test_in_spawn_process(run_io_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
