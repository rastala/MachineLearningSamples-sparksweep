# Parameter Sweep on Spark

This sample shows how to leverage the [_spark-sklearn_ package](https://spark-packages.org/package/databricks/spark-sklearn) in parallel to perform parameter sweep on Spark worker nodes, and compare that with the regular parameter sweep performed in scikit-learn.

It also shows how to add [published Spark packages](https://spark-packages.org/) to your project using the _Spark_dependencies.yml_ file in the _aml_config_ folder.

## Create Compute Targets

First, create compute targets for running the parameter sweep: Docker and optionally, HDInsight Spark Cluster. Select **File**, **Open Command Prompt** and enter following commands to create the compute targets.

Prepare Docker image. Note that this step may take several minutes:

```azurecli
$ az ml experiment prepare -c docker
```

Create HDInsight Spark Cluster using [these instructions](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql), and attach it as compute target by using:

```azurecli
$ az ml computecontext attach --name <myhdi> --address <ssh-myhdi.azurehdinsight.net> --username <sshusername> --password <sshpwd> --type cluster
```

## Add Spark Package Dependencies

This project is configured to install spark-sklearn package when running on Docker or HDInsight Spark cluster. To view how this is done, or to change or add dependencies, edit the __aml_config/Spark_dependencies.yml__ file.

```
configuration: {}
repositories:
  - "https://mmlspark.azureedge.net/maven"
  - "https://spark-packages.org/packages"
packages:
  - group: "com.microsoft.ml.spark"
    artifact: "mmlspark_2.11"
    version: "0.7.91"
  - group: "databricks"
    artifact: "spark-sklearn"
    version: "0.2.0"
```

## Run the Model

**Note**: To run sweep_spark.py, you must select Docker or HDInsight Spark cluster as compute target. The sweep_sklearn.py can run on local Python also.

Local run using scikit-learn:

```azurecli
$ az ml experiment submit -c local .\sweep_sklearn.py
```

Docker run using spark-sklearn

```azurecli
$ az ml experiment submit -c docker .\sweep_spark.py
```

HDInsight Spark run using spark-sklearn:

```azurecli
$ az ml experiment submit -c <myhdi> .\sweep_spark.py
```
