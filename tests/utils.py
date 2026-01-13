import os
import sys
from contextlib import contextmanager
from multiprocessing import Manager, get_context


@contextmanager
def suppress_output(stdout=True, stderr=True):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout if stdout else None
        old_stderr = sys.stderr if stderr else None

        if stdout:
            sys.stdout = devnull
        if stderr:
            sys.stderr = devnull

        try:
            yield
        finally:
            if stdout:
                sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr


def _worker_process(func_name, module_name, config, queue):
    """Worker function to run in spawned process - must be at module level for pickling"""
    import traceback

    # Don't capture stdout/stderr - let C++ errors go directly to console
    # This allows us to see SIGABRT messages from the C++ library

    try:
        print(f"Worker process starting - Python: {sys.version}", file=sys.stderr)
        print(f"Worker CWD: {os.getcwd()}", file=sys.stderr)
        print(
            f"DFTRACER_ENABLE: {os.environ.get('DFTRACER_ENABLE', 'NOT SET')}",
            file=sys.stderr,
        )

        import importlib

        module = importlib.import_module(module_name)
        test_func = getattr(module, func_name)

        print(f"Successfully imported {module_name}.{func_name}", file=sys.stderr)
        sys.stderr.flush()

        result = test_func(config)
        sys.stdout.flush()

        print("Test completed successfully", file=sys.stderr)
        sys.stderr.flush()

        queue.put(
            {
                "success": result,
                "stdout": "",
                "stderr": "",
                "error": None,
            }
        )
    except Exception as e:
        sys.stderr.flush()

        queue.put(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
    except BaseException as e:
        sys.stderr.flush()

        queue.put(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"BaseException: {str(e)}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def run_test_in_spawn_process(test_func, test_config, timeout=120):
    """
    Run a test function in a separate process using multiprocessing spawn.

    Args:
        test_func: The test function to run (must be a module-level function)
        test_config: Configuration dictionary to pass to the test function
        timeout: Timeout in seconds (default: 120)

    Returns:
        None (raises AssertionError if test fails)

    Raises:
        TimeoutError: If the test times out
        RuntimeError: If the process fails to return results
        AssertionError: If the test fails
    """

    # Use spawn context to ensure fresh process
    ctx = get_context("spawn")

    # Create a managed queue to ensure proper sharing across spawned processes
    manager = Manager()
    output_queue = manager.Queue()

    func_name = test_func.__name__
    module_name = test_func.__module__

    # Extract environment variables from config
    env_vars = test_config.get("env", {})

    # Temporarily modify parent environment before spawning
    # With spawn, child inherits parent's environment at spawn time
    old_env = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create and start process - it will inherit the modified environment
        process = ctx.Process(
            target=_worker_process,
            args=(func_name, module_name, test_config, output_queue),
        )
        process.start()

        # Wait for completion with timeout
        process.join(timeout=timeout)
    finally:
        # Restore original environment in parent
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    # Check if process is still alive (timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(
            f"Test {test_config.get('name', 'unknown')} timed out after {timeout} seconds"
        )

    # Check exit code first
    if process.exitcode != 0:
        # Try to get error info from queue if available
        error_msg = f"Test {test_config.get('name', 'unknown')} process exited with code {process.exitcode}"
        try:
            if not output_queue.empty():
                result_data = output_queue.get(timeout=1)
                if result_data.get("error"):
                    error_msg += f"\nError: {result_data['error']}"
                if result_data.get("traceback"):
                    error_msg += f"\nTraceback:\n{result_data['traceback']}"
                if result_data.get("stderr"):
                    error_msg += f"\nStderr:\n{result_data['stderr']}"
        except Exception:
            # Ignore errors when trying to get additional error info
            pass
        raise RuntimeError(error_msg)

    # Get results from queue
    try:
        result_data = output_queue.get(timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Test {test_config.get('name', 'unknown')} process completed but failed to return results. "
            f"Exit code: {process.exitcode}"
        ) from e

    print(f"Test {test_config.get('name', 'unknown')} process output:")
    print(result_data["stdout"])
    if result_data["stderr"]:
        print(f"Test {test_config.get('name', 'unknown')} process errors:")
        print(result_data["stderr"])

    if result_data["error"]:
        print(f"Test failed with error: {result_data['error']}")
        if "traceback" in result_data:
            print(result_data["traceback"])
        raise AssertionError(
            f"Test {test_config.get('name', 'unknown')} failed in spawned process: {result_data['error']}"
        )

    assert result_data["success"], (
        f"Test {test_config.get('name', 'unknown')} failed in spawned process"
    )


def validate_log_files(log_file, test_name, expected_count=0, mode="exact"):
    """
    Validate DFTracer log files and return event count.

    Args:
        log_file: Path to the expected log file
        test_name: Name of the test (for error messages)
        expected_count: Expected number of events (0 means just check > 5 for "exact" mode,
                       or minimum count for "at_least" mode)
        mode: Validation mode - "exact" (default) or "at_least"
              - "exact": Event count must exactly match expected_count (or > 5 if expected_count=0)
              - "at_least": Event count must be >= expected_count

    Returns:
        Tuple of (event_count, cpp_library_available)
    """
    import glob

    event_count = 0
    cpp_library_available = True

    # Check if C++ library is available
    try:
        import dftracer.dftracer as cpp_libs  # noqa: F401

        print("dftracer C++ library is available")
    except ImportError:
        cpp_library_available = False
        print("dftracer C++ library is NOT available - tests will run in no-op mode")

    # Check DFTRACER_ENABLE
    DFTRACER_ENABLE = os.environ.get("DFTRACER_ENABLE", "0") == "1"

    if DFTRACER_ENABLE:
        # Try different log file patterns
        log_pattern = log_file.replace(".pfw", "*-app.pfw")
        log_files = glob.glob(log_pattern)

        if not log_files and os.path.exists(log_file):
            log_files = [log_file]

        if not log_files:
            log_pattern_simple = log_file.replace(".pfw", "*.pfw")
            log_files = glob.glob(log_pattern_simple)

        print(f"Looking for log files with pattern: {log_pattern}")
        print(f"Found log files: {log_files}")

        for log_file_path in log_files:
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path) as f:
                        lines = f.readlines()
                        event_count += len(lines)
                    print(f"Found {len(lines)} events in {log_file_path}")
                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")

        if event_count == 0:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir):
                log_dir_files = os.listdir(log_dir)
                print(f"Files in log directory: {log_dir_files}")
            else:
                print(f"Log directory does not exist: {log_dir}")

    print(f"Test {test_name} completed with {event_count} events")

    # Assertions
    print(f"Expected event count: {expected_count} (mode: {mode})")
    if not DFTRACER_ENABLE:
        assert event_count == 0, (
            f"Expected 0 events when DFTRACER_ENABLE=0 but got {event_count} for test {test_name}"
        )
    elif not cpp_library_available:
        assert event_count == 0, (
            f"Expected 0 events when C++ library not available but got {event_count} for test {test_name}"
        )
    elif mode == "at_least":
        # For "at_least" mode, event_count must be >= expected_count
        if expected_count > 0:
            assert event_count >= expected_count, (
                f"Expected at least {expected_count} events but got {event_count} for test {test_name}"
            )
        else:
            assert event_count > 5, (
                f"Expected at least some events (> 5) but got {event_count} for test {test_name}"
            )
    elif mode == "exact":
        # For "exact" mode, event_count must exactly match expected_count
        if expected_count > 0:
            assert event_count == expected_count, (
                f"Expected exactly {expected_count} events but got {event_count} for test {test_name}"
            )
        else:
            assert event_count > 5, (
                f"Expected some events (> 5) but got {event_count} for test {test_name}"
            )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'exact' or 'at_least'")

    return event_count, cpp_library_available


def get_dftracer_preload_path():
    """Find the path to libdftracer_preload.so in the virtual environment"""
    try:
        import site

        # Get all site-packages directories
        site_packages_dirs = site.getsitepackages()

        # Also check the user site-packages if it exists
        try:
            site_packages_dirs.append(site.getusersitepackages())
        except AttributeError:
            pass

        # Check for dftracer_libs in each site-packages directory
        for site_dir in site_packages_dirs:
            dftracer_libs_path = os.path.join(site_dir, "dftracer")

            if os.path.isdir(dftracer_libs_path):
                # Try both possible locations
                lib64_path = os.path.join(
                    dftracer_libs_path, "lib64", "libdftracer_preload.so"
                )
                lib_path = os.path.join(
                    dftracer_libs_path, "lib", "libdftracer_preload.so"
                )

                if os.path.exists(lib64_path):
                    return lib64_path
                elif os.path.exists(lib_path):
                    return lib_path

        print("Warning: dftracer_libs directory or libdftracer_preload.so not found")
        return ""

    except Exception as e:
        print(f"Warning: Error finding dftracer_libs: {e}")
        return ""
