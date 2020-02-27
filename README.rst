====
CIML
====

ciml (CI machine learning) is a machine learning pipeline for analyzing CI
results.

Getting Started
---------------

To use ciml, you have to collect a dataset and train the model. Once that's done
you can perform predictions on new data using the MQTT triggered pipeline.

The complete workflow to create a trained model is designed to allow easy
experimentation and tuning of data preparation as well as model parameters and
hyper-parameters:

* Cache raw data (dstat CSV files) and run metadata from OpenStack log server
  and `test result database<https://docs.openstack.org/infra/system-config/logstash.html#subunit2sql>`. After this step, everything else may be done
  offline, as only cached data will be used
* Build a normalized dataset, split into training, dev and test. This step
  allows to define the number of samples, the level of downsampling (in any)
  and which dstat columns we want to use
* Define an experiment. Experiments are defined by a model with its
  parameters and hyper-parameters. Experiments are associated to datasets.
  Each dataset can have multiple experiments
* Train a model i.e. run an experiment against a dataset


To cache raw data locally, use `ciml-cache-data`:

.. code:: shell

  ciml-cache-data --build-name <build-name> --db-uri <db-uri>

Example:
.. code::shell
  ciml-cache-data --build-name tempest-full --db-uri mysql+pymysql://query:query@logstash.openstack.org/subunit2sql

This connects to the OpenStack `subunit2sql<https://docs.openstack.org/subunit2sql/latest/reference/index.html>` database, fetch all runs that
match the specified build_name and try to download the dstat data from
logs.openstack.org. The dstat file and build results are stored gzipped
under data/.raw and data/.metadata.

Running ciml-cache-data with the same build name again extends the cache.
Data for different build names can be cached, dedicated JSON files keep track
of which data belongs to which build.

For more help run:

.. code:: shell

  ciml-cache-data --help

To create a normalized dataset, use `ciml-build-dataset`:

.. code:: shell

  ciml-build-dataset --dataset <dataset-name> --build-name <build-name>

This selects the required number of runs from the local cache, it loads the
data into memory, it runs the normalization step and it saves the three
resulting datasets (traning, dev and test) into numpy compressed archives
(.npz) files. It also creates one compressed archive for the feature labels.

There are several extra options that can be used to control size and
normalization of the resulting dataset. For more help run:

.. code:: shell

  ciml-build-dataset --help

It is also possible to visualize the lenght of example prior to normalization,
and how it maps to the example class (0 for passed, 1 for failed).
To produce the graph, run:

.. code:: shell

    ciml-build-dataset --dataset <dataset-name> --visualize

.. image:: sizes_by_result.png

To define an experiment, use `ciml-setup-experiment`:

.. code:: shell

  ciml-setup-experiment --dataset <dataset-name> --experiment <exp-name> \
    --estimator (tf.contrib.learn.SVM|tf.estimator.DNNClassifier)

This stores the specified model, parameter and hyper-parameters into a JSON
file in a dedicated folder, which is going to host TensorFlow model files as
well.

Full more help run:

.. code:: shell

  ciml-setup-experiment --help

To run training, use `ciml-train-model`:

.. code:: shell

  ciml-train-model --dataset <dataset-name> --experiment <exp-name>

This loads the dataset from the numpy compressed archives, it initialize the
model based on the experiment settings and runs training against the training
set and evaluation against the test set.

To use TensorBoard for a specific experiment, run:

.. code:: shell

  tensorboard --logdir <ciml_base_path>/data/<dataset-name>/<experiment-name>

To start the MQTT triggered pipeline, and make predictions on new data, use:

.. code:: shell

  ciml-mqtt-trainer
