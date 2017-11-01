# Policy Gradient Implimentation (WIP)

1. vanilla policy gradient
1. ppo


## Build the project

1. download maven
1. Build code with maven
   1. go to directory analytics-zoo/apps/policyGradient/
   1. type command in terminal: ```mvn clean package```
1. you'll find 2 jars and 1 zip file under analytics-zoo/apps/policyGradient/target
   * ```rl-0.1-SNAPSHOT.jar```
   * ```rl-0.1-SNAPSHOT-jar-with-dependencies.jar```
   * ```rl-0.1-SNAPSHOT-python-api.zip```


## Use with Pyspark + Jupyter

* Learn how to work with Pyspark and Jupyter first: refer to https://bigdl-project.github.io/master/#PythonUserGuide/run-without-pip/#run-from-pyspark-jupyter

* add the extra jars into below options in script  
   1. --py-files
   1. --jars
   1. --conf spark.driver.extraClassPath
   1. --conf spark.executor.extraClassPath=

Example
```bash
${SPARK_HOME}/bin/pyspark \
        --master ${MASTER} \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --driver-cores 4  \
       --driver-memory 64g  \
       --total-executor-cores 8  \
       --executor-cores 1  \
       --executor-memory 64g \
       --conf spark.akka.frameSize=64 \
       --conf spark.sql.catalogImplementation='' \
       --py-files ${PYTHON_API_ZIP_PATH},${RLlib_PYTHON_API_ZIP_PATH} \
       --jars ${BigDL_JAR_PATH},${RLlib_JAR_PATH} \
       --conf spark.driver.extraClassPath=${BigDL_JAR_PATH}:${RLlib_JAR_PATH} \
       --conf spark.executor.extraClassPath=${BigDL_HOME}/dist/lib/${BigDL_lib}-jar-with-dependencies.jar:${RLlib_JAR_PATH}

```
 
