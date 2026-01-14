.. pydftracer documentation master file

Welcome to pydftracer documentation!
====================================

**pydftracer** is the Python frontend for `DFTracer <https://dftracer.readthedocs.io/>`_,
a powerful I/O profiling and tracing tool. This library provides Python bindings and utilities
to integrate DFTracer profiling capabilities into Python applications, with specialized support
for AI/ML frameworks and PyTorch integration.

Features
--------

- **Python Frontend**: Easy-to-use Python API for DFTracer profiler
- **Type-Safe Decorators**: Fully type-checked with mypy, preserves function signatures
- **Function Decorators**: Simple ``@dft_fn`` decorator for tracing Python functions
- **AI/ML Support**: Specialized tracing for AI/ML workflows
- **PyTorch Integration**: Capture PyTorch profiler events to DFTracer traces
- **PyTorch Dynamo Integration**: Wrapper of PyTorch's Dynamo
- **Automatic I/O Tracing**: Transparent tracing of I/O operations when enabled
- **Debugging Tools**: Built-in debugging utilities
- **Environment Configuration**: Flexible configuration via environment variables
- **Cross-platform**: Works on Linux and other Unix-like systems

.. toctree::
   :maxdepth: 1
   :caption: Links:

   DFTracer Documentation <https://dftracer.readthedocs.io/>
   DFTracer GitHub <https://github.com/LLNL/dftracer>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples
   type_safety
   ai_ml_guide
   pytorch
   dynamo_guide
   developers
   api/index

Getting Started
---------------

To get started with pydftracer, check out the :doc:`installation` guide
and then follow the :doc:`quickstart` tutorial.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pydftracer

For more detailed installation instructions, see :doc:`installation`.

Quick Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Enable DFTracer via environment variable
   export DFTRACER_ENABLE=1

.. code-block:: python

   from dftracer.python import dftracer, dft_fn

   # Initialize the DFTracer logger
   df_logger = dftracer.initialize_log("trace.pfw", "/tmp/data", -1)

   # Create a tracer for your functions
   io_tracer = dft_fn("io_operations")

   @io_tracer.log
   def read_data(filename):
       with open(filename, 'r') as f:
           return f.read()

   # I/O operations will be automatically profiled
   data = read_data('data.txt')

   # Finalize the logger
   df_logger.finalize()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
