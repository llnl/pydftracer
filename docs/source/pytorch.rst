PyTorch Profiler Integration
=============================

This guide explains how to use pydftracer with PyTorch's built-in profiler
to capture profiling events and log them to DFTracer trace files.

Overview
--------

The PyTorch Profiler integration allows you to:

- Capture PyTorch profiler events (CPU, CUDA, memory usage)
- Log profiling data to DFTracer trace files with category ``PP`` (PyTorch Profiler)
- Combine PyTorch profiling with I/O tracing for comprehensive analysis

Setup
-----

Enable DFTracer before using the PyTorch Profiler integration:

.. code-block:: bash

   export DFTRACER_ENABLE=1

Basic Usage
-----------

Use the ``trace_handler`` function as the ``on_trace_ready`` callback for PyTorch's profiler:

.. code-block:: python

   import torch
   from torch.profiler import profile, schedule, ProfilerActivity
   from dftracer.python import dftracer
   from dftracer.python.torch import trace_handler

   # Initialize DFTracer logger
   df_logger = dftracer.initialize_log("profiler_trace.pfw", None, -1)

   # Define profiler schedule
   profiler_schedule = schedule(
       wait=1,
       warmup=1,
       active=2,
       repeat=1,
   )

   # Run profiler with trace_handler
   with profile(
       activities=[ProfilerActivity.CPU],
       schedule=profiler_schedule,
       on_trace_ready=trace_handler,
       profile_memory=True,
       with_stack=True,
   ) as p:
       for step in range(4):
           # Your training code here
           model(input_data)
           p.step()

   df_logger.finalize()

Training Loop Example
---------------------

Complete example with a training loop:

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.profiler import profile, schedule, ProfilerActivity, record_function
   from dftracer.python import dftracer
   from dftracer.python import dft_fn as Profile
   from dftracer.python.torch import trace_handler

   # Initialize logger
   df_logger = dftracer.initialize_log("training.pfw", None, -1)

   # Model setup
   model = nn.Linear(10, 2)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   criterion = nn.CrossEntropyLoss()

   # Create DFTracer profile for additional tracing
   df_test = Profile("training")

   @df_test.log
   def training_step(inputs, labels):
       optimizer.zero_grad()

       with record_function("forward"):
           outputs = model(inputs)

       with record_function("loss"):
           loss = criterion(outputs, labels)

       loss.backward()
       optimizer.step()
       return loss.item()

   # Profiler schedule
   profiler_schedule = schedule(wait=1, warmup=1, active=2, repeat=1)

   with profile(
       activities=[ProfilerActivity.CPU],
       schedule=profiler_schedule,
       on_trace_ready=trace_handler,
       profile_memory=True,
   ) as p:
       for step in range(4):
           inputs = torch.randn(8, 10)
           labels = torch.randint(0, 2, (8,))
           loss = training_step(inputs, labels)
           p.step()

   df_logger.finalize()

What Gets Traced
----------------

The ``trace_handler`` captures the following from PyTorch profiler events:

- **Event name**: Operation or kernel name
- **Timing**: Start time and duration (microseconds)
- **Device type**: CPU or CUDA device
- **Memory usage**: CPU and device memory
- **Input shapes**: Size of input tensors
- **CPU/Device utilization**: Percentage metrics

All events are logged with category ``PP`` (PyTorch Profiler) in the trace file.

Example Output
--------------

Trace entries from PyTorch Profiler look like:

.. code-block:: json

   {"name":"aten::linear","cat":"PP","ts":1234567890,"dur":150,"args":{"device":0,"cpu_memory":1024,"device_memory_usage":0}}

Combining with I/O Tracing
--------------------------

To capture both PyTorch profiler events and I/O operations:

.. code-block:: bash

   export DFTRACER_ENABLE=1
   export DFTRACER_DATA_DIR=all

.. code-block:: python

   from dftracer.python import dftracer, ai
   from dftracer.python.torch import trace_handler
   from torch.profiler import profile, ProfilerActivity

   df_logger = dftracer.initialize_log("combined.pfw", None, -1)

   # Use AI decorators for I/O tracing
   @ai.data.item
   def load_data(idx):
       # Your data loading code
       return data

   # Use PyTorch profiler for compute tracing
   with profile(
       activities=[ProfilerActivity.CPU],
       on_trace_ready=trace_handler,
   ) as p:
       for step in range(steps):
           data = load_data(step)
           output = model(data)
           p.step()

   df_logger.finalize()

API Reference
-------------

.. py:function:: dftracer.python.torch.trace_handler(profiler_result)

   Callback function for PyTorch profiler's ``on_trace_ready`` parameter.

   :param profiler_result: PyTorch profiler result object containing events
   :type profiler_result: torch.profiler.profiler.ProfilerResult

   The handler iterates through all profiler events and logs them to DFTracer
   with the following information:

   - ``name``: Event key/name
   - ``cat``: Always ``"PP"`` (PyTorch Profiler)
   - ``start_time``: Event start time in microseconds
   - ``duration``: Event duration in microseconds
   - ``int_args``: device, cpu_memory, is_remote, device_memory_usage, input_size
   - ``float_args``: total_cpu_percent, total_device_percent
