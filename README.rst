====
CIML
====

ciml (CI machine learning) is a machine learning pipeline for analyzing CI
results.

Getting Started
---------------

To use ciml, you have to collect a dataset and train the model. Once that's done
you can perform predictions on new data using the MQTT triggered pipeline.

To collect a dataset, use ciml-build-dataset:

.. code:: shell

  ciml-build-dataset --dataset <dataset-name> --build-name <build_name>

This connects to the OpenStack subunit2sql database, fetch all runs that
match the specified build_name and try to download the dstat data from
logs.openstack.org. The dstat file and build results are stored gzipped
under data/<dataset-name>/raw.

Running ciml-build-dataset with the same dataset again extends the dataset if
new results are found.

Full more help run:

.. code:: shell

  ciml-build-dataset --help


To train the model, ciml-build-dataset:

.. code:: shell

  ciml-train-model --dataset <dataset-name>

This looks in the raw data folder and loads the list of runs (examples) from
there along with the test results (classes: pass or failed).
Data is normalised and then passed to the trainer for train the model.

Full more help run:

.. code:: shell

  ciml-train-model --help

It is also possible to visualize the lenght of example prior to normalization,
and how it maps to the example class (0 for passed, 1 for failed).
To produce the graph, run:

.. code:: shell

  ciml-train-model --dataset <dataset-name> --no-train --visualize

.. image:: sizes_by_result.png


To start the MQTT triggered pipeline, and make predictions on new data, use:

.. code:: shell

  ciml-mqtt-trainer
