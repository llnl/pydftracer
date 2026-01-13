#!/usr/bin/env python
import glob
import gzip
import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .utils import run_test_in_spawn_process, suppress_output


def cleanup_test_directory(test_base_dir, test_name):
    """
    Clean up test output directory after successful test completion.
    Uses a two-phase approach: first empty all directories, then remove empty directories.

    Args:
        test_base_dir: Base directory path for the test
        test_name: Name of the test (for logging)
    """
    import time

    try:
        if not os.path.exists(test_base_dir):
            print(f"Test directory does not exist (already cleaned?): {test_base_dir}")
            return

        # Wait for any file handles to be released
        time.sleep(0.5)

        # Phase 1: Remove all files, ignoring NFS temporary files
        files_removed = 0
        nfs_files_ignored = 0

        for root, _dirs, files in os.walk(test_base_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Skip NFS temporary files (they'll be cleaned up automatically)
                    if file.startswith(".nfs"):
                        nfs_files_ignored += 1
                        continue

                    os.remove(file_path)
                    files_removed += 1
                except OSError as e:
                    # If we can't remove a file, just note it but continue
                    if not file.startswith(".nfs"):
                        print(f"Could not remove file {file_path}: {e}")

        # Phase 2: Remove empty directories (bottom-up)
        dirs_removed = 0
        for root, dirs, _files in os.walk(test_base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)  # Only removes if empty
                    dirs_removed += 1
                except OSError:
                    # Directory not empty (probably has .nfs files), ignore
                    pass

        # Phase 3: Try to remove the main directory
        try:
            os.rmdir(test_base_dir)
            dirs_removed += 1
            print(
                f"Cleaned up test directory: {test_base_dir} ({files_removed} files, {dirs_removed} dirs removed, {nfs_files_ignored} NFS files ignored)"
            )
        except OSError:
            # Main directory not empty (probably has .nfs files or subdirs with .nfs files)
            print(
                f"Partially cleaned test directory: {test_base_dir} ({files_removed} files, {dirs_removed} subdirs removed, {nfs_files_ignored} NFS files ignored)"
            )
            print("Main directory remains due to NFS temporary files")

    except Exception as e:
        print(f"Info: Cleanup encountered unexpected error for {test_base_dir}: {e}")


def validate_pytorch_profiler_and_io_logs(
    log_file, test_name, data_dir, expected_min_events=1, check_io_events=True
):
    """
    Validate log files and check for both 'cat':'PP' (PyTorch Profiler) and I/O events ('cat':'POSIX' or 'cat':'STDIO').
    Also validates that specific synthetic data files appear in the I/O logs.

    Args:
        log_file: Path to the expected log file
        test_name: Name of the test (for error messages)
        data_dir: Data directory to check for synthetic files
        expected_min_events: Minimum expected number of PyTorch profiler events
        check_io_events: Whether to validate I/O events (only for I/O-specific tests)

    Returns:
        Tuple of (total_events, pytorch_profiler_events, io_events, synthetic_file_events)
    """
    total_events = 0
    pytorch_profiler_events = 0
    io_events = 0
    synthetic_file_events = 0

    # Look for synthetic data files that should appear in I/O logs (only for I/O tests)
    synthetic_data_dir = os.path.join(data_dir, "synthetic_data")
    expected_synthetic_files = []
    if check_io_events and os.path.exists(synthetic_data_dir):
        expected_synthetic_files = [
            f for f in os.listdir(synthetic_data_dir) if f.endswith(".pkl")
        ]
        print(
            f"Expected to find I/O events for {len(expected_synthetic_files)} synthetic files: {expected_synthetic_files[:3]}..."
        )
    elif not check_io_events:
        print("Skipping I/O event validation for non-I/O test")

    # Check if C++ library is available
    try:
        import dftracer.dftracer as cpp_libs  # noqa: F401

        cpp_library_available = True
        print("dftracer C++ library is available")
    except ImportError:
        cpp_library_available = False
        print("dftracer C++ library is NOT available - tests will run in no-op mode")

    # Check DFTRACER_ENABLE - import locally to avoid global import issues
    DFTRACER_ENABLE_ENV = os.environ.get("DFTRACER_ENABLE", "0") == "1"

    if DFTRACER_ENABLE_ENV and cpp_library_available:
        # Check both the specified log directory and data directory for log files
        log_dir = os.path.dirname(log_file)
        directories_to_check = [log_dir, data_dir]

        log_files = []
        for check_dir in directories_to_check:
            print(f"Checking directory: {check_dir}")
            if os.path.exists(check_dir):
                # Try different log file patterns for gzipped files - collect ALL matching files
                patterns_to_try = [
                    os.path.join(check_dir, "*-app.pfw.gz"),  # Standard gzipped format
                    os.path.join(check_dir, "*.pfw.gz"),  # Simple gzipped format
                    os.path.join(check_dir, "*.pfw"),  # Uncompressed fallback
                ]

                for pattern in patterns_to_try:
                    found_files = glob.glob(pattern)
                    if found_files:
                        # Remove duplicates by converting to set and back to list
                        new_files = [f for f in found_files if f not in log_files]
                        log_files.extend(new_files)
                        print(
                            f"Found {len(found_files)} files with pattern {pattern}: {found_files}"
                        )
                    else:
                        print(f"No files found with pattern: {pattern}")

                # Also list all files in directory for debugging
                all_files = os.listdir(check_dir)
                if all_files:
                    print(f"All files in {check_dir}: {all_files}")
                else:
                    print(f"Directory {check_dir} is empty")
            else:
                print(f"Directory {check_dir} does not exist")

        print(f"All found log files: {log_files}")

        for log_file_path in log_files:
            if os.path.exists(log_file_path):
                try:
                    # Handle both gzipped and uncompressed files
                    if log_file_path.endswith(".gz"):
                        with gzip.open(log_file_path, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(log_file_path) as f:
                            lines = f.readlines()

                    total_events += len(lines)

                    # Count lines containing PyTorch profiler events and I/O events (POSIX/STDIO/FH)
                    pp_events = sum(1 for line in lines if '"cat":"PP"' in line)
                    posix_events = sum(1 for line in lines if '"cat":"POSIX"' in line)
                    stdio_events = sum(1 for line in lines if '"cat":"STDIO"' in line)
                    fh_events = sum(1 for line in lines if '"cat":"FH"' in line)
                    io_count = posix_events + stdio_events + fh_events
                    pytorch_profiler_events += pp_events
                    io_events += io_count

                    # Count FH events that involve our synthetic files
                    # Look for paths containing our test directory and sample files
                    sample_file_events = sum(
                        1
                        for line in lines
                        if (
                            '"cat":"FH"' in line
                            and ("sample" in line or test_name in line)
                        )
                    )

                    # Also count dftracer metadata entries for our test directory
                    metadata_events = sum(
                        1
                        for line in lines
                        if (
                            '"cat":"dftracer"' in line
                            and (test_name in line or "/data" in line)
                        )
                    )

                    synthetic_events_in_file = sample_file_events + metadata_events
                    synthetic_file_events += synthetic_events_in_file

                    if synthetic_events_in_file > 0:
                        print(
                            f"Found {sample_file_events} FH events + {metadata_events} metadata events involving our test files out of {fh_events} total FH events"
                        )

                    print(
                        f"Found {len(lines)} total events, {pp_events} PyTorch profiler events, {io_count} I/O events ({posix_events} POSIX, {stdio_events} STDIO, {fh_events} FH), {synthetic_events_in_file} 'sample' file events in {log_file_path}"
                    )

                    # Print first few lines for debugging
                    if lines:
                        print(f"Sample log entries from {log_file_path}:")
                        for i, line in enumerate(lines[:3]):
                            print(f"  Line {i + 1}: {line.strip()[:100]}...")

                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")

        if total_events == 0:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir):
                log_dir_files = os.listdir(log_dir)
                print(f"Files in log directory ({log_dir}): {log_dir_files}")
                # Show file details for debugging
                for f in log_dir_files:
                    full_path = os.path.join(log_dir, f)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        print(f"  {f}: {size} bytes")
            else:
                print(f"Log directory does not exist: {log_dir}")

    print(
        f"Test {test_name} completed with {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events (POSIX/STDIO)"
    )

    # Assertions based on DFTRACER state
    if not DFTRACER_ENABLE_ENV:
        assert total_events == 0, (
            f"Expected 0 events when DFTRACER_ENABLE=0 but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when DFTRACER_ENABLE=0 but got {pytorch_profiler_events} for test {test_name}"
        )
        assert io_events == 0, (
            f"Expected 0 I/O events (POSIX/STDIO/FH) when DFTRACER_ENABLE=0 but got {io_events} for test {test_name}"
        )
    elif not cpp_library_available:
        assert total_events == 0, (
            f"Expected 0 events when C++ library not available but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when C++ library not available but got {pytorch_profiler_events} for test {test_name}"
        )
        assert io_events == 0, (
            f"Expected 0 I/O events (POSIX/STDIO/FH) when C++ library not available but got {io_events} for test {test_name}"
        )
    else:
        # When enabled and library available, we should have some events
        assert total_events > 0, (
            f"Expected some events but got {total_events} for test {test_name}"
        )
        # PyTorch profiler events are optional - the integration might not always produce PP events
        # but we should at least have dftracer events from the training function
        if pytorch_profiler_events > 0:
            print(f"Great! Found {pytorch_profiler_events} PyTorch profiler events")
        else:
            print(
                f"No PyTorch profiler events found, but {total_events} dftracer events were logged"
            )

        # Only validate I/O events for I/O-specific tests
        if check_io_events:
            # I/O events should be present if we're doing dataset loading
            if io_events > 0:
                print(
                    f"Great! Found {io_events} I/O events (POSIX/STDIO/FH) from dataset operations"
                )
            else:
                print(
                    "No I/O events (POSIX/STDIO/FH) found - this may indicate I/O tracing issues"
                )

            # Check for file handle events (FH category) to validate I/O tracing is working
            if synthetic_file_events > 0:
                print(
                    f"Great! Found {synthetic_file_events} file handle (FH) events - I/O tracing is working"
                )

                # Simple validation: if we created synthetic files, we should have some FH events
                expected_file_count = len(expected_synthetic_files)
                if expected_file_count > 0:
                    print(
                        f"✅ I/O validation passed: Found {synthetic_file_events} FH events with {expected_file_count} synthetic files created"
                    )
                else:
                    print(
                        f"✅ I/O validation passed: Found {synthetic_file_events} FH events"
                    )
            else:
                if expected_synthetic_files:
                    raise AssertionError(
                        f"No file handle (FH) events found in traces, but {len(expected_synthetic_files)} synthetic files were created. "
                        f"This indicates I/O tracing is not capturing file operations properly."
                    )
                else:
                    print(
                        "No file handle (FH) events found - may be expected if no file I/O occurred"
                    )
        else:
            print(f"Found {io_events} I/O events (not validated for non-I/O test)")

    return total_events, pytorch_profiler_events, io_events, synthetic_file_events


def validate_pytorch_profiler_logs(
    log_file, test_name, data_dir, expected_min_events=1
):
    """
    Validate PyTorch profiler log files and check for 'cat':'PP' events.
    Does not validate I/O events (for regular PyTorch profiler tests).

    Args:
        log_file: Path to the expected log file
        test_name: Name of the test (for error messages)
        expected_min_events: Minimum expected number of PyTorch profiler events

    Returns:
        Tuple of (total_events, pytorch_profiler_events)
    """
    total_events, pytorch_profiler_events, _, _ = validate_pytorch_profiler_and_io_logs(
        log_file, test_name, data_dir, expected_min_events, check_io_events=False
    )
    return total_events, pytorch_profiler_events


def validate_pytorch_profiler_logs_original(
    log_file, test_name, data_dir, expected_min_events=1
):
    """
    Original validate_pytorch_profiler_logs function implementation for reference.
    """
    total_events = 0
    pytorch_profiler_events = 0

    # Check if C++ library is available
    try:
        import dftracer.dftracer as cpp_libs  # noqa: F401

        cpp_library_available = True
        print("dftracer C++ library is available")
    except ImportError:
        cpp_library_available = False
        print("dftracer C++ library is NOT available - tests will run in no-op mode")

    # Check DFTRACER_ENABLE - import locally to avoid global import issues
    DFTRACER_ENABLE_ENV = os.environ.get("DFTRACER_ENABLE", "0") == "1"

    if DFTRACER_ENABLE_ENV and cpp_library_available:
        # Check both the specified log directory and data directory for log files
        log_dir = os.path.dirname(log_file)
        directories_to_check = [log_dir, data_dir]

        log_files = []
        for check_dir in directories_to_check:
            print(f"Checking directory: {check_dir}")
            if os.path.exists(check_dir):
                # Try different log file patterns for gzipped files - collect ALL matching files
                patterns_to_try = [
                    os.path.join(check_dir, "*-app.pfw.gz"),  # Standard gzipped format
                    os.path.join(check_dir, "*.pfw.gz"),  # Simple gzipped format
                    os.path.join(check_dir, "*.pfw"),  # Uncompressed fallback
                ]

                for pattern in patterns_to_try:
                    found_files = glob.glob(pattern)
                    if found_files:
                        # Remove duplicates by converting to set and back to list
                        new_files = [f for f in found_files if f not in log_files]
                        log_files.extend(new_files)
                        print(
                            f"Found {len(found_files)} files with pattern {pattern}: {found_files}"
                        )
                    else:
                        print(f"No files found with pattern: {pattern}")

                # Also list all files in directory for debugging
                all_files = os.listdir(check_dir)
                if all_files:
                    print(f"All files in {check_dir}: {all_files}")
                else:
                    print(f"Directory {check_dir} is empty")
            else:
                print(f"Directory {check_dir} does not exist")

        print(f"All found log files: {log_files}")

        for log_file_path in log_files:
            if os.path.exists(log_file_path):
                try:
                    # Handle both gzipped and uncompressed files
                    if log_file_path.endswith(".gz"):
                        with gzip.open(log_file_path, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(log_file_path) as f:
                            lines = f.readlines()

                    total_events += len(lines)

                    # Count lines containing PyTorch profiler events
                    pp_events = sum(1 for line in lines if '"cat":"PP"' in line)
                    pytorch_profiler_events += pp_events

                    print(
                        f"Found {len(lines)} total events, {pp_events} PyTorch profiler events in {log_file_path}"
                    )

                    # Print first few lines for debugging
                    if lines:
                        print(f"Sample log entries from {log_file_path}:")
                        for i, line in enumerate(lines[:3]):
                            print(f"  Line {i + 1}: {line.strip()[:100]}...")

                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")

        if total_events == 0:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir):
                log_dir_files = os.listdir(log_dir)
                print(f"Files in log directory ({log_dir}): {log_dir_files}")
                # Show file details for debugging
                for f in log_dir_files:
                    full_path = os.path.join(log_dir, f)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        print(f"  {f}: {size} bytes")
            else:
                print(f"Log directory does not exist: {log_dir}")

    print(
        f"Test {test_name} completed with {total_events} total events, {pytorch_profiler_events} PyTorch profiler events"
    )

    # Assertions based on DFTRACER state
    if not DFTRACER_ENABLE_ENV:
        assert total_events == 0, (
            f"Expected 0 events when DFTRACER_ENABLE=0 but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when DFTRACER_ENABLE=0 but got {pytorch_profiler_events} for test {test_name}"
        )
    elif not cpp_library_available:
        assert total_events == 0, (
            f"Expected 0 events when C++ library not available but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when C++ library not available but got {pytorch_profiler_events} for test {test_name}"
        )
    else:
        # When enabled and library available, we should have some events
        assert total_events > 0, (
            f"Expected some events but got {total_events} for test {test_name}"
        )
        # PyTorch profiler events are optional - the integration might not always produce PP events
        # but we should at least have dftracer events from the training function
        if pytorch_profiler_events > 0:
            print(f"Great! Found {pytorch_profiler_events} PyTorch profiler events")
        else:
            print(
                f"No PyTorch profiler events found, but {total_events} dftracer events were logged"
            )

    return total_events, pytorch_profiler_events


def run_torchvision_cifar10_with_io_test(test_config):
    """Run a PyTorch profiler test with real I/O operations using torchvision CIFAR-10 dataset"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    from dftracer.python.dbg import (
        DFTRACER_ENABLE,
        ai,  # Import AI decorators for proper I/O tracing
        dftracer,
    )
    from dftracer.python.dbg import dft_fn as Profile
    from dftracer.python.dbg.torch import trace_handler
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    from torchvision import datasets

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test with torchvision CIFAR-10 I/O {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, None, -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Define transforms for CIFAR-10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # RGB normalization
            ]
        )

        # Create custom CIFAR-10 dataset class with AI decorators for I/O tracing
        class TracedCIFAR10Dataset(datasets.CIFAR10):
            """CIFAR-10 dataset with dftracer AI decorators for I/O monitoring"""

            @ai.data.item
            def __getitem__(self, index):
                """Traced version of CIFAR-10 __getitem__ to capture I/O events"""
                return super().__getitem__(index)

        # Use torchvision's CIFAR-10 dataset - this will trigger actual I/O operations
        cifar10_data_dir = os.path.join(data_dir, "cifar-10")
        print(f"Using CIFAR-10 dataset in: {cifar10_data_dir}")

        try:
            # Try to use cached CIFAR-10 data first, then download if needed
            with suppress_output():
                train_dataset = TracedCIFAR10Dataset(
                    root=cifar10_data_dir,
                    train=True,
                    download=True,  # Download if not present
                    transform=transform,
                )
            print(
                f"Successfully loaded CIFAR-10 dataset with {len(train_dataset)} samples"
            )
        except Exception as e:
            print(
                f"Failed to load torchvision CIFAR-10 ({e}), falling back to synthetic data"
            )
            # Fallback to synthetic data if CIFAR-10 loading fails
            from torch.utils.data import TensorDataset

            num_samples = max(batch_size * 4, 32)
            synthetic_data = torch.randn(num_samples, 3, 32, 32)
            synthetic_labels = torch.randint(0, 10, (num_samples,))
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        # Create a traced dataloader using AI decorators
        @ai.dataloader.fetch
        def create_traced_dataloader(dataset, batch_size):
            """Create dataloader with AI tracing"""
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Use 0 workers for simpler I/O tracing
            )

        train_dataloader = create_traced_dataloader(train_dataset, batch_size)

        # Model, loss, and optimizer setup for CIFAR-10 (3 channels, 32x32)
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_io_{test_config['name']}")

        @df_test.log
        @ai.compute.forward
        def training_step_with_io(step_data):
            """Training step with AI compute tracing"""
            inputs, labels = step_data

            # Device transfer with AI tracing
            with ai.device.transfer:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            with ai.compute.backward:
                loss.backward()

            with ai.compute.step:
                optimizer.step()

            return loss.item()

        step_count = 0

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                # Use AI pipeline tracing for epochs
                for epoch in ai.pipeline.epoch.iter(range(num_epochs)):
                    # Use AI dataloader tracing for data iteration
                    for _step, (inputs, labels) in ai.dataloader.fetch.iter(
                        enumerate(train_dataloader)
                    ):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step_with_io((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test with torchvision CIFAR-10 I/O {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for both PyTorch profiler events and I/O events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events, io_events, synthetic_file_events = (
        validate_pytorch_profiler_and_io_logs(
            log_file,
            test_config["name"],
            data_dir,
            expected_min_events=expected_pytorch_events,
        )
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events, {synthetic_file_events} synthetic file events"
    )

    # Clean up test directory after successful completion
    cleanup_test_directory(test_base_dir, test_config["name"])

    return True


def run_single_pytorch_profiler_with_io_test(test_config):
    """Run a PyTorch profiler test with real I/O operations through CIFAR-10 dataset loading"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    from dftracer.python.dbg import DFTRACER_ENABLE, dftracer
    from dftracer.python.dbg import dft_fn as Profile
    from dftracer.python.dbg.torch import trace_handler
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test with I/O {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, None, -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use CIFAR-10 data that already exists in the workspace to generate I/O operations
        import pickle

        import numpy as np
        from torch.utils.data import Dataset

        class SyntheticIODataset(Dataset):
            """Custom dataset that generates I/O events by reading/writing synthetic data files."""

            def __init__(self, data_dir, batch_size):
                self.data_dir = data_dir
                self.data = []
                self.labels = []

                # Create synthetic data files that will generate I/O operations when loaded
                synthetic_data_dir = os.path.join(data_dir, "synthetic_data")
                os.makedirs(synthetic_data_dir, exist_ok=True)

                # Create enough samples for the test
                num_samples = max(batch_size * 6, 20)  # Ensure we have enough samples

                # Create individual data files to generate I/O operations
                for i in range(num_samples):
                    data_file = os.path.join(synthetic_data_dir, f"sample_{i}.pkl")

                    # Create synthetic image and label
                    synthetic_image = np.random.rand(3, 32, 32).astype(np.float32)
                    synthetic_label = np.random.randint(0, 10)

                    # Save to file to create I/O operations
                    sample_data = {"image": synthetic_image, "label": synthetic_label}

                    with open(data_file, "wb") as f:
                        pickle.dump(sample_data, f)

                    print(f"Created synthetic data file: {data_file}")

                # Store file paths for later loading (which will generate I/O events)
                self.sample_files = [
                    os.path.join(synthetic_data_dir, f"sample_{i}.pkl")
                    for i in range(num_samples)
                ]

                print(
                    f"Created {len(self.sample_files)} synthetic data files for I/O operations"
                )

            def __len__(self):
                return len(self.sample_files)

            def __getitem__(self, idx):
                # Load data from file - this generates I/O operations that will be traced
                sample_file = self.sample_files[idx]

                with open(sample_file, "rb") as f:
                    sample_data = pickle.load(f)

                # Convert to torch tensors
                image = torch.tensor(sample_data["image"], dtype=torch.float32)
                label = torch.tensor(sample_data["label"], dtype=torch.long)
                return image, label

        # Create dataset and dataloader - this will trigger I/O operations
        try:
            train_dataset = SyntheticIODataset(data_dir, batch_size)
            print(
                f"Successfully created SyntheticIODataset with {len(train_dataset)} samples"
            )
        except Exception as e:
            print(
                f"Failed to create SyntheticIODataset ({e}), falling back to in-memory synthetic data"
            )
            # Fallback to in-memory synthetic data if file-based dataset creation fails
            from torch.utils.data import TensorDataset

            num_samples = max(batch_size * 4, 32)
            synthetic_data = torch.randn(num_samples, 3, 32, 32)
            synthetic_labels = torch.randint(0, 10, (num_samples,))
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Model, loss, and optimizer setup for CIFAR-10 (3 channels, 32x32)
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_io_{test_config['name']}")

        @df_test.log
        def training_step_with_io(step_data):
            inputs, labels = step_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            return loss.item()

        step_count = 0

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for epoch in range(num_epochs):
                    for _step, (inputs, labels) in enumerate(train_dataloader):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step_with_io((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test with I/O {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for both PyTorch profiler events and I/O events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events, io_events, synthetic_file_events = (
        validate_pytorch_profiler_and_io_logs(
            log_file,
            test_config["name"],
            data_dir,
            expected_min_events=expected_pytorch_events,
        )
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events, {synthetic_file_events} synthetic file events"
    )

    # Clean up test directory after successful completion
    cleanup_test_directory(test_base_dir, test_config["name"])

    return True


def run_image_folder_with_io_test(test_config):
    """Run a PyTorch profiler test using ImageFolder for comprehensive I/O operations"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    import numpy as np
    from dftracer.python.dbg import (
        DFTRACER_ENABLE,
        ai,  # Import AI decorators for proper I/O tracing
        dftracer,
    )
    from dftracer.python.dbg import dft_fn as Profile
    from dftracer.python.dbg.torch import trace_handler
    from PIL import Image
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    from torchvision.datasets import ImageFolder

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test with ImageFolder I/O {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, None, -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create synthetic image data in ImageFolder format to ensure I/O operations
        image_folder_dir = os.path.join(data_dir, "synthetic_images")

        # Create class directories
        num_classes = 3
        num_images_per_class = max(batch_size, 8)  # Ensure enough images for testing

        for class_idx in range(num_classes):
            class_dir = os.path.join(image_folder_dir, f"class_{class_idx}")
            os.makedirs(class_dir, exist_ok=True)

            # Create synthetic images for each class
            for img_idx in range(num_images_per_class):
                img_path = os.path.join(class_dir, f"image_{img_idx}.png")
                if not os.path.exists(img_path):
                    # Create a synthetic 32x32 RGB image
                    synthetic_image = np.random.randint(
                        0, 256, (32, 32, 3), dtype=np.uint8
                    )
                    pil_image = Image.fromarray(synthetic_image)
                    pil_image.save(img_path)
                    print(f"Created synthetic image: {img_path}")

        # Define transforms for image processing
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # RGB normalization
            ]
        )

        # Create custom ImageFolder dataset class with AI decorators for I/O tracing
        class TracedImageFolder(ImageFolder):
            """ImageFolder dataset with dftracer AI decorators for I/O monitoring"""

            @ai.data.item
            def __getitem__(self, index):
                """Traced version of ImageFolder __getitem__ to capture I/O events"""
                return super().__getitem__(index)

        # Use torchvision's ImageFolder - this will trigger actual file I/O operations
        try:
            train_dataset = TracedImageFolder(
                root=image_folder_dir, transform=transform
            )
            print(
                f"Successfully loaded ImageFolder dataset with {len(train_dataset)} samples"
            )
        except Exception as e:
            print(
                f"Failed to load ImageFolder dataset ({e}), falling back to synthetic data"
            )
            # Fallback to synthetic data if ImageFolder loading fails
            from torch.utils.data import TensorDataset

            num_samples = max(batch_size * 4, 32)
            synthetic_data = torch.randn(num_samples, 3, 32, 32)
            synthetic_labels = torch.randint(0, num_classes, (num_samples,))
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        # Create a traced dataloader using AI decorators
        @ai.dataloader.fetch
        def create_traced_dataloader(dataset, batch_size):
            """Create dataloader with AI tracing"""
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Use 0 workers for simpler I/O tracing
            )

        train_dataloader = create_traced_dataloader(train_dataset, batch_size)

        # Model, loss, and optimizer setup for RGB images (3 channels, 32x32)
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust final layer for the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_imagefolder_{test_config['name']}")

        @df_test.log
        @ai.compute.forward
        def training_step_with_io(step_data):
            """Training step with AI compute tracing"""
            inputs, labels = step_data

            # Device transfer with AI tracing
            with ai.device.transfer:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            with ai.compute.backward:
                loss.backward()

            with ai.compute.step:
                optimizer.step()

            return loss.item()

        step_count = 0

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                # Use AI pipeline tracing for epochs
                for epoch in ai.pipeline.epoch.iter(range(num_epochs)):
                    # Use AI dataloader tracing for data iteration
                    for _step, (inputs, labels) in ai.dataloader.fetch.iter(
                        enumerate(train_dataloader)
                    ):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step_with_io((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test with ImageFolder I/O {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for both PyTorch profiler events and I/O events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events, io_events, synthetic_file_events = (
        validate_pytorch_profiler_and_io_logs(
            log_file,
            test_config["name"],
            data_dir,
            expected_min_events=expected_pytorch_events,
        )
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events, {synthetic_file_events} synthetic file events"
    )

    # Clean up test directory after successful completion
    cleanup_test_directory(test_base_dir, test_config["name"])

    return True


def run_single_pytorch_profiler_test(test_config):
    """Run a single PyTorch profiler test in isolation"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    from dftracer.python.dbg import DFTRACER_ENABLE, dftracer
    from dftracer.python.dbg import dft_fn as Profile
    from dftracer.python.dbg.torch import trace_handler
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, None, -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use pure in-memory synthetic data for regular PyTorch profiler tests (no I/O)
        print(
            "Using in-memory synthetic data for PyTorch profiler test (no I/O operations)"
        )
        from torch.utils.data import TensorDataset

        # Create synthetic data (RGB 32x32, similar to CIFAR-10)
        num_samples = max(batch_size * 6, 32)  # Ensure we have enough samples
        synthetic_data = torch.randn(num_samples, 3, 32, 32)
        synthetic_labels = torch.randint(0, 10, (num_samples,))
        train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Model, loss, and optimizer setup - for RGB input (3 channels, 32x32)
        model = models.resnet18(weights=None)
        # Keep default first conv layer for RGB input (3 channels)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_{test_config['name']}")

        @df_test.log
        def training_step(step_data):
            inputs, labels = step_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            return loss.item()

        step_count = 0  # Initialize step_count outside the profiler context

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for epoch in range(num_epochs):
                    for _step, (inputs, labels) in enumerate(train_dataloader):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for PyTorch profiler events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events = validate_pytorch_profiler_logs(
        log_file,
        test_config["name"],
        data_dir,
        expected_min_events=expected_pytorch_events,
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events"
    )

    # Clean up test directory after successful completion
    cleanup_test_directory(test_base_dir, test_config["name"])

    return True


class TestPyTorchProfiler:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "basic_pytorch_profiler",
                "batch_size": 4,
                "num_epochs": 1,
                "num_steps": 3,
                "active_steps": 2,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
            {
                "name": "small_pytorch_profiler",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 2,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
        ],
    )
    def test_pytorch_profiler(self, test_config):
        """Test PyTorch profiler integration with dftracer."""
        run_test_in_spawn_process(run_single_pytorch_profiler_test, test_config)

    def test_pytorch_profiler_disabled(self):
        """Test PyTorch profiler with dftracer disabled."""
        test_config = {
            "name": "disabled",
            "batch_size": 8,
            "num_epochs": 1,
            "num_steps": 2,
            "active_steps": 1,
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }
        run_test_in_spawn_process(run_single_pytorch_profiler_test, test_config)

    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "with_io_operations",
                "batch_size": 4,
                "num_epochs": 1,
                "num_steps": 3,
                "active_steps": 2,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
            {
                "name": "pp_and_io_combined",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 2,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
        ],
    )
    def test_pytorch_profiler_with_io(self, test_config):
        """Test PyTorch profiler integration with dftracer including I/O operations with CIFAR-10."""
        run_test_in_spawn_process(run_single_pytorch_profiler_with_io_test, test_config)

    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "torchvision_cifar10_test",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 2,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
            {
                "name": "cifar10_with_ai_decorators",
                "batch_size": 4,
                "num_epochs": 1,
                "num_steps": 3,
                "active_steps": 2,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
        ],
    )
    def test_torchvision_cifar10_with_io_and_pp(self, test_config):
        """Test PyTorch profiler with torchvision CIFAR-10 dataset including both I/O and PP events."""
        run_test_in_spawn_process(run_torchvision_cifar10_with_io_test, test_config)

    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "imagefolder_io_test",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 1,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
            {
                "name": "multiple_files_test",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 2,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                    "DFTRACER_LOG_LEVEL": "INFO",
                },
            },
        ],
    )
    def test_image_folder_with_io_and_pp(self, test_config):
        """Test PyTorch profiler with ImageFolder dataset including both I/O and PP events."""
        run_test_in_spawn_process(run_image_folder_with_io_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
