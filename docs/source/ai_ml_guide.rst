AI/ML Tracing Guide
===================

This guide covers how to use pydftracer's specialized AI/ML tracing features to profile
deep learning and machine learning workflows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Motivation
----------

Since DFTracer's release, we've successfully traced numerous AI/DL pipelines.
However, analysis revealed that the resulting traces differ widely across workloads.

This inconsistency is largely due to varied naming schemes used by different users.
Even when the intent is similar, the lack of a standard makes it hard to build analysis tools
that work reliably across use cases.

This API introduces consistent annotation conventions to help users instrument their code more uniformly.
With these standards in place, tools like `DFAnalyzer <https://github.com/LLNL/dfanalyzer>`_ can
operate more effectively — they will *just work™*, reducing fatigue for researchers and
developers analyzing AI/DL workloads.

Overview
--------

The ``ai`` module provides decorators and context managers for tracing common AI/ML operations:

- **Data operations**: Loading, preprocessing, and augmentation
- **Dataloader**: Batch fetching and iteration
- **Device operations**: Data transfer to/from GPU
- **Compute operations**: Forward pass, backward pass, optimization steps
- **Communication**: Distributed training operations (all_reduce, etc.)
- **Checkpointing**: Model save/load operations
- **Pipeline**: Training/validation/test loops
- **Other**: Any I/O or logging activity that does not fit a standard AI/ML category

Basic Setup
-----------

First, enable DFTracer and initialize the logger:

.. code-block:: bash

   export DFTRACER_ENABLE=1

.. code-block:: python

   from dftracer.python import dftracer, ai
   import numpy as np

   # Initialize logger
   df_logger = dftracer.initialize_log("ai_trace.pfw", "/tmp/data", -1)

   # Your AI/ML code here

   # Finalize when done
   df_logger.finalize()

Data Operations
---------------

Tracing Data Loading
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai
   import numpy as np

   class IOHandler:
       @ai.data.item
       def read(self, filename: str):
           return np.load(filename)

       def write(self, filename: str, data):
           with open(filename, "wb") as f:
               np.save(f, data)

   io = IOHandler()
   data = io.read("data.npy")  # This read will be traced

Dataloader Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   @ai.dataloader.fetch
   def read_batch(data_dir: str, num_files: int):
       for i in range(num_files):
           yield io.read(f"{data_dir}/{i}.npy")

   # Iterate over batches with tracing
   for step, data in ai.dataloader.fetch.iter(enumerate(read_batch("/data", 100))):
       # Process data
       pass

Data Preprocessing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   @ai.data.preprocess.derive(name="collate")
   def collate(data):
       # Collate batch data
       return data

   @ai.data.preprocess.derive(name="augment")
   def augment(data):
       # Apply data augmentation
       return data

   # Use in your pipeline
   processed_data = collate(raw_data)
   augmented_data = augment(processed_data)

Device Operations
-----------------

Tracing GPU Transfers
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   @ai.device.transfer
   def transfer_to_gpu(data):
       # Transfer data to GPU
       # In real code: return data.cuda()
       return data

   # Traced transfer
   gpu_data = transfer_to_gpu(cpu_data)

Compute Operations
------------------

Forward and Backward Passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   @ai.compute.forward
   def forward(model, data):
       return model(data)

   @ai.compute.backward
   def backward(loss):
       loss.backward()

   # Use in training loop
   output = forward(model, batch)
   loss = criterion(output, labels)
   backward(loss)

Optimization Steps
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   class Hook:
       def before_step(self):
           ai.compute.step.start()

       def after_step(self):
           ai.compute.step.stop()

   hook = Hook()

   # In training loop
   hook.before_step()
   # ... forward, backward, optimizer.step()
   hook.after_step()

Communication Tracing
---------------------

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   @ai.compute.backward
   def backward_with_sync():
       loss.backward()
       # Trace distributed communication
       with ai.comm.all_reduce():
           # All-reduce gradients
           pass

   # Can also disable tracing for specific operations
   with ai.comm.all_reduce(enable=False):
       # This won't be traced
       pass

Checkpointing
-------------

Model Checkpoints
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai
   from time import sleep

   class Checkpoint:
       @ai.checkpoint.init
       def __init__(self):
           # Initialize checkpoint system
           pass

       @ai.checkpoint.capture
       def save(self, state):
           # Save model checkpoint
           return state

       @ai.checkpoint.restart
       def load(self, checkpoint_path):
           # Load model checkpoint
           return {}

   checkpoint = Checkpoint()
   checkpoint.load("checkpoint.pt")
   # ... training ...
   checkpoint.save({"model": model.state_dict()})

I/O Operations
--------------

The ``io`` sub-tracer is available on both ``data`` and ``checkpoint`` to annotate
the four fundamental I/O phases of any data object lifecycle:

- **open** — opening or loading a data object (e.g. opening a file, loading a dataset)
- **read** — reading data from an already-open object
- **write** — writing data into an object
- **close** — closing or deleting a data object

This lets you precisely attribute time to each I/O phase rather than lumping it
into a coarser ``data.item`` or ``checkpoint.capture`` span.

Tracing Data I/O
~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   class DatasetReader:
       @ai.data.io.open
       def open(self, path: str):
           self._file = open(path, "rb")

       @ai.data.io.read
       def read(self, n: int):
           return self._file.read(n)

       @ai.data.io.close
       def close(self):
           self._file.close()

   # Or with context managers
   def load_sample(path: str):
       with ai.data.io.open:
           fh = open(path, "rb")
       with ai.data.io.read:
           data = fh.read()
       with ai.data.io.close:
           fh.close()
       return data

Tracing Checkpoint I/O
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai
   import torch

   class CheckpointManager:
       @ai.checkpoint.io.open
       def _open(self, path: str):
           return open(path, "rb")

       @ai.checkpoint.io.read
       def load(self, path: str):
           with self._open(path) as f:
               return torch.load(f)

       @ai.checkpoint.io.write
       def save(self, state: dict, path: str):
           torch.save(state, path)

       @ai.checkpoint.io.close
       def delete(self, path: str):
           import os
           os.remove(path)

Combined with capture/restart context:

.. code-block:: python

   from dftracer.python import ai

   def save_checkpoint(model, path: str):
       with ai.checkpoint.capture:
           with ai.checkpoint.io.open:
               f = open(path, "wb")
           with ai.checkpoint.io.write:
               torch.save(model.state_dict(), f)
           with ai.checkpoint.io.close:
               f.close()

   def load_checkpoint(path: str):
       with ai.checkpoint.restart:
           with ai.checkpoint.io.open:
               f = open(path, "rb")
           with ai.checkpoint.io.read:
               state = torch.load(f)
           with ai.checkpoint.io.close:
               f.close()
       return state

Other Operations
----------------

Use the ``other`` category to annotate any I/O or function call that does not belong to a standard
AI/ML category — for example, writing application logs, calling an external REST API, reading
configuration files, or any ad-hoc file access outside the data pipeline.

Tracing Other I/O
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   class ConfigReader:
       @ai.other.io.open
       def open(self, path: str):
           self._f = open(path, "r")

       @ai.other.io.read
       def read(self):
           return self._f.read()

       @ai.other.io.close
       def close(self):
           self._f.close()

   # Or with context managers
   with ai.other.io.open:
       f = open("config.json", "r")
   with ai.other.io.read:
       cfg = f.read()
   with ai.other.io.close:
       f.close()

Tracing Log / API Calls
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ai.other.log`` to annotate logging writes, metric emissions, or external API calls:

.. code-block:: python

   from dftracer.python import ai

   @ai.other.log
   def emit_metric(name: str, value: float):
       # Write to external monitoring system
       requests.post("https://metrics.example.com", json={name: value})

   # Or as a context manager
   with ai.other.log:
       logger.info("Training step complete")

   # Or derive a named sub-tracer
   api_call = ai.other.log.derive(name="rest_api")

   @api_call
   def fetch_config(url: str):
       return requests.get(url).json()

Training Pipeline
-----------------

Complete Training Loop
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import dftracer, ai
   import numpy as np

   # Initialize logger
   df_logger = dftracer.initialize_log("training.pfw", "/tmp/data", -1)

   @ai.pipeline.train
   def train(num_epochs, num_batches):
       # Training loop with epoch tracing
       for epoch in ai.pipeline.epoch.iter(range(num_epochs)):
           for step, data in ai.dataloader.fetch.iter(range(num_batches)):
               # Update current step and epoch
               ai.update(step=step, epoch=epoch)

               # Data loading
               batch = load_batch(step)

               # Transfer to device
               batch = transfer(batch)

               # Forward pass
               output = forward(model, batch)

               # Backward pass
               backward(loss)

   train(num_epochs=5, num_batches=100)
   df_logger.finalize()

Metadata and Custom Tags
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dftracer.python import ai

   # Start/stop epochs with metadata logging
   for epoch in range(num_epochs):
       ai.pipeline.epoch.start(metadata=True)

       # Training code
       for step in range(num_steps):
           ai.update(step=step, epoch=epoch)
           # ... training code ...

       ai.pipeline.epoch.stop(metadata=True)

Advanced Features
-----------------

Custom Categories
~~~~~~~~~~~~~~~~~

You can create custom AI tracers with specific categories:

.. code-block:: python

   from dftracer.python import DFTracerAI

   # Create custom AI tracer
   custom_tracer = DFTracerAI(
       cat="custom_category",
       name="my_operation",
       epoch=1,
       step=100,
       enable=True
   )

Disabling Specific Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can selectively disable tracing for specific AI categories programmatically:

.. code-block:: python

   from dftracer.python import ai

   # Disable all AI tracing
   ai.disable()

   # Or disable specific categories
   ai.dataloader.disable()
   ai.device.disable()
   ai.compute.disable()
   ai.comm.disable()
   ai.checkpoint.disable()
   ai.other.disable()

AI/DL Logging Conventions
--------------------------

We define six main categories of logging. Each category, along with its subcategories (children),
is implemented as a wrapper around ``dft_fn``. This means you can use these categories in your
codebase the same way you would use ``dft_fn`` directly.

.. list-table:: AI/DL Logging Conventions
   :widths: 15 15 30 40
   :header-rows: 1

   * - Category
     - Name
     - Access Path
     - Description
   * - Compute
     - Forward
     - ``ai.compute.forward``
     - Forward pass of the network
   * -
     - Backward
     - ``ai.compute.backward``
     - Backward pass / gradient computation
   * -
     - Step
     - ``ai.compute.step``
     - Optimizer step (parameter update)
   * - Data
     - Preprocess
     - ``ai.data.preprocess``
     - Dataset-level preprocessing
   * -
     - Item
     - ``ai.data.item``
     - Per-item transformation or loading
   * -
     - IO: Open
     - ``ai.data.io.open``
     - Open or load a data object
   * -
     - IO: Read
     - ``ai.data.io.read``
     - Read data from an open object
   * -
     - IO: Write
     - ``ai.data.io.write``
     - Write data to an object
   * -
     - IO: Close
     - ``ai.data.io.close``
     - Close or delete a data object
   * - DataLoader
     - Fetch
     - ``ai.dataloader.fetch``
     - Fetch a batch from DataLoader
   * - Comm
     - Send
     - ``ai.comm.send``
     - Point-to-point send
   * -
     - Receive
     - ``ai.comm.receive``
     - Point-to-point receive
   * -
     - Barrier
     - ``ai.comm.barrier``
     - Synchronization barrier
   * -
     - Broadcast
     - ``ai.comm.bcast``
     - Broadcast (one-to-many)
   * -
     - Reduce
     - ``ai.comm.reduce``
     - Reduce (many-to-one)
   * -
     - All-Reduce
     - ``ai.comm.all_reduce``
     - All-reduce (many-to-many)
   * -
     - Gather
     - ``ai.comm.gather``
     - Gather (many-to-one)
   * -
     - All-Gather
     - ``ai.comm.all_gather``
     - All-gather (many-to-many)
   * -
     - Scatter
     - ``ai.comm.scatter``
     - Scatter (one-to-many)
   * -
     - Reduce-Scatter
     - ``ai.comm.reduce_scatter``
     - Reduce-scatter (many-to-many)
   * -
     - All-to-All
     - ``ai.comm.all_to_all``
     - All-to-all (many-to-many)
   * - Device
     - Transfer
     - ``ai.device.transfer``
     - Host-to-device or device-to-host memory transfer
   * - Checkpoint
     - Capture
     - ``ai.checkpoint.capture``
     - Capture a model checkpoint
   * -
     - Restart
     - ``ai.checkpoint.restart``
     - Restart from a model checkpoint
   * -
     - IO: Open
     - ``ai.checkpoint.io.open``
     - Open or load a checkpoint file
   * -
     - IO: Read
     - ``ai.checkpoint.io.read``
     - Read checkpoint data from an open file
   * -
     - IO: Write
     - ``ai.checkpoint.io.write``
     - Write checkpoint data to a file
   * -
     - IO: Close
     - ``ai.checkpoint.io.close``
     - Close or delete a checkpoint file
   * - Pipeline
     - Epoch
     - ``ai.pipeline.epoch``
     - An entire training or evaluation epoch
   * -
     - Train
     - ``ai.pipeline.train``
     - Training phase
   * -
     - Evaluate
     - ``ai.pipeline.evaluate``
     - Evaluation or validation phase
   * -
     - Test
     - ``ai.pipeline.test``
     - Testing or inference phase
   * - Other
     - Log
     - ``ai.other.log``
     - Logging writes, metric emissions, or external API calls
   * -
     - IO: Open
     - ``ai.other.io.open``
     - Open a file or resource outside the data pipeline
   * -
     - IO: Read
     - ``ai.other.io.read``
     - Read from an open resource
   * -
     - IO: Write
     - ``ai.other.io.write``
     - Write to a resource
   * -
     - IO: Close
     - ``ai.other.io.close``
     - Close or release a resource

Flexible API Styles
-------------------

DFTracer AI Logging provides flexible APIs to match different coding styles.
You can use decorators, context managers, or iterable wrappers.

Decorator Style
~~~~~~~~~~~~~~~

**Without arguments** — use it directly to wrap a function:

.. code-block:: python

   @ai.compute.forward
   def forward(model, x):
       loss = model(x)
       return loss

**With arguments** — pass metadata to the event:

.. code-block:: python

   @ai.compute.forward(args={"arg1": "value1", "arg2": "value2"})
   def forward(model, x):
       loss = model(x)
       return loss

Context Manager Style
~~~~~~~~~~~~~~~~~~~~~

Use it to wrap blocks of code inside a ``with`` statement:

**Without arguments:**

.. code-block:: python

   with ai.compute.forward:
       loss = model(x)

**With arguments:**

.. code-block:: python

   with ai.compute.forward(args={"arg1": "value1", "arg2": "value2"}):
       loss = model(x)

Iterable Style
~~~~~~~~~~~~~~

You can also wrap iterators like data loaders:

.. code-block:: python

   for batch in ai.dataloader.fetch.iter(dataloader):
       # Process batch
       pass

Constructor Hooking
~~~~~~~~~~~~~~~~~~~

You can annotate constructors directly using category-specific hooks:

.. code-block:: python

   class MyDataset:
       @ai.data.item.init  # special `init` event for this category
       def __init__(self, ...):
           # Initialization logic
           pass

Updating Arguments
------------------

Every profiler (like ``ai.compute.forward``) provides an ``update`` method to
dynamically change metadata. These updates apply to the entire subtree of that event.

.. code-block:: python

   @ai.compute.forward
   def forward(model, x):
       loss = model(x)
       return loss

   for epoch in ai.pipeline.epoch.iter(range(num_epochs)):
       for step, batch in ai.dataloader.fetch.iter(enumerate(dataloader)):
           # Update metadata for the current context
           ai.compute.forward.update(epoch=epoch, step=step)
           forward(model, batch)

Force Enable or Disable Specific Events
----------------------------------------

You can override the global or category-level logging state for individual events
by setting the ``enable`` flag explicitly.

.. code-block:: python

   ai.compute.disable()  # Disable all compute events

   @ai.compute.forward(enable=True)  # Force-enable this specific event
   def forward(model, x):
       loss = model(x)
       return loss

   with ai.compute.backward(enable=True):  # Force-enable this block
       loss.backward()

   ai.compute.enable()  # Enable all compute events

   @ai.compute.forward(enable=False)  # Force-disable this one
   def forward(model, x):
       loss = model(x)
       return loss

Hook/Checkpoint Style
---------------------

For scenarios where you can't use decorators or context managers directly
(e.g., TensorFlow SessionHook), you can manually call profiler methods:

.. code-block:: python

   class DFTracerProfilingHook(tf.train.SessionRunHook):
       def begin(self):
           self._global_step_tensor = training_util._get_or_create_global_step_read()
           if self._global_step_tensor is None:
               raise RuntimeError("Global step should be created.")
           ai.pipeline.epoch.start()

       def end(self, session):
           ai.pipeline.epoch.stop()

       def before_run(self, run_context):
           global_step = run_context.session.run(self._global_step_tensor)
           ai.update(step=global_step)
           ai.compute.start()

       def after_run(self, run_context, run_values):
           ai.compute.stop()

Derivation
----------

You can derive new profilers from existing ones for more dynamic logging.
The derived profiler becomes a child of the original profiler, inheriting its context.

.. code-block:: python

   class Dataset:
       def __getitem__(self, idx: int):
           data = ...
           with ai.data.preprocess:
               # Process data
               pass
           return data

   # This becomes name="preprocess.collate" with cat="data"
   @ai.data.preprocess.derive(name="collate")
   def collate(batch):
       return batch

   # Or (context-manager style)
   profiler_collate = ai.data.preprocess.derive(name="collate")

   def collate_fn(batch):
       with profiler_collate:
           return collate(batch)

   # Update derived profiler
   profiler_collate.update(epoch=epoch)

   # This also updates all children of ai.data.preprocess
   ai.data.preprocess.update(epoch=epoch)

Metadata / Streaming Style
---------------------------

By default, DFTracer logs events with a start and end time (duration-based).
For real-time monitoring, use ``metadata=True`` to log events immediately:

.. code-block:: python

   # Regular mode
   for epoch in ai.pipeline.epoch.iter(range(num_epochs)):
       for step in range(num_steps):
           # Do work
           pass

   # Metadata mode
   for epoch in range(num_epochs):
       ai.pipeline.epoch.start(metadata=True)
       for step in range(num_steps):
           # Do work
           pass
       ai.pipeline.epoch.stop(metadata=True)

**Regular mode output:**

.. code-block:: json

   {"id":27,"name":"epoch.block","cat":"pipeline","pid":2877353,"tid":2877353,
    "ts":1753123213646764,"dur":828765,"ph":"X",
    "args":{"hhash":"2a702c695247d487","p_idx":6,"count":"1","level":2}}

**Metadata mode output:**

.. code-block:: json

   {"id":6,"name":"CM","cat":"dftracer","pid":2876815,"tid":2876815,"ph":"M",
    "args":{"hhash":"2a702c695247d487","name":"epoch.end","value":"1753123070219202"}}
   {"id":6,"name":"CM","cat":"dftracer","pid":2876815,"tid":2876815,"ph":"M",
    "args":{"hhash":"2a702c695247d487","name":"epoch.start","value":"1753123070219648"}}

Init Events
-----------

Log initialization phases using the ``init`` method:

.. code-block:: python

   class Checkpoint:
       @ai.checkpoint.init
       def __init__(self):
           # Initialize something
           pass

   # Or
   with ai.checkpoint.init:
       # Initialize something
       pass

**Output:**

.. code-block:: json

   {"id":7,"name":"checkpoint.init","cat":"checkpoint","pid":444541,"tid":444541,
    "ts":1753136835509693,"dur":100583,"ph":"X",
    "args":{"hhash":"2a702c695247d487","p_idx":6,"level":2}}

Caveats
-------

Call Ordering Matters
~~~~~~~~~~~~~~~~~~~~~

The order of calls affects whether events get logged.

**This works:**

.. code-block:: python

   class Checkpoint:
       @ai.checkpoint.init  # Instance tracked internally
       def __init__(self):
           pass

   if __name__ == "__main__":
       ai.checkpoint.disable()  # Disables all checkpoint events

**This doesn't work as expected:**

.. code-block:: python

   class Checkpoint:
       @ai.checkpoint.init()  # Parentheses create instance immediately
       def __init__(self):
           pass

   if __name__ == "__main__":
       ai.checkpoint.disable()  # Can't affect already-created instance

**Solutions:**

1. Use the decorator without parentheses, or call ``disable()`` before defining your class
2. Only use parentheses ``()`` when you need to force enable/disable a specific event
3. To add metadata, use the ``update()`` method instead
4. To create variations of an event, use the ``derive()`` method instead

Summary
-------

The AI/ML tracing features in pydftracer provide:

- **Structured tracing** for common ML operations
- **Hierarchical** tracking of training loops
- **Minimal overhead** with automatic profiling
- **Flexible** decorator-based API
- **Multiple usage patterns** (decorators, context managers, iterables)
- **Dynamic configuration** (enable/disable, metadata updates)
- **Integration** with existing ML code

For complete API reference, see :doc:`api/ai`.
