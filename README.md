
## Building Spark

Spark is built using [Apache Maven](http://maven.apache.org/).
To build Spark and its example programs, run:

    build/mvn -DskipTests clean package


You can set the MASTER environment variable when running examples to submit
examples to a cluster. This can be a mesos:// or spark:// URL,
"yarn" to run on YARN, and "local" to run
locally with one thread, or "local[N]" to run locally with N threads. You
can also use an abbreviated class name if the class is in the `examples`
package. For instance:

    MASTER=spark://host:7077 ./bin/run-example SparkPi

Many of the example programs print usage help if no params are given.
# Spark Source Code Compilation, Cluster Deployment, and IDEA Integration

This README provides a comprehensive guide on how to compile the Spark source code, deploy a Spark cluster, and integrate Spark development in the IDEA environment. It covers the necessary steps and configurations for a seamless Spark development and deployment process.

## 1. Introduction
This guide is designed to assist you in setting up a Spark development and deployment environment. It details the process of compiling the Spark source code, deploying a Spark cluster, and integrating Spark with the IDEA IDE for efficient development. The steps are based on Spark 3.2.0 and have been tested in a specific environment.

## 2. Source Code Download
1. Open the Apache Spark™ website: [Apache Spark™ - Unified Engine for large-scale data analytics](https://spark.apache.org/).
2. Navigate to the downloads page and select the source code option. You can choose a specific Spark release (e.g., 3.2.0).
3. Download the source code (e.g., `spark-3.2.0.tgz`).
4. Upload the downloaded source code to the `/spark` directory on your Hadoop server (e.g., Hadoop01).
5. Use the command `tar -zxvf spark-3.2.0.tgz` to extract the source code in the `/spark` directory.

## 3. Source Code Compilation
### 3.1 Prerequisites Installation
- **Maven**:
    1. Open the Maven website: [Maven – Welcome to Apache Maven](http://maven.apache.org/).
    2. Navigate to the downloads page and select the latest binary package (e.g., apache-maven-3.8.4-bin.tar.gz).
    3. On the Hadoop server (e.g., Hadoop01), create a `/maven` directory and download the Maven package using the command `curl -O https://dlcdn.apache.org/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz`.
    4. Extract the downloaded package using `tar -zxvf apache-maven-3.8.4-bin.tar.gz`.
    5. Move the contents of the extracted directory to make `/maven` the Maven home directory.
    6. Configure the Maven environment variables in the `~/.bash_profile` file:
```bash
# User specific environment and startup programs
PATH=$PATH:$HOME/bin
JAVA_HOME=/java
HADOOP_HOME=/hadoop
MAVEN_HOME=/maven
PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$MAVEN_HOME/bin
export JAVA_HOME
export HADOOP_HOME
export MAVEN_HOME
export PATH
```
    7. Use the command `source ~/.bash_profile` to make the environment variables take effect.
    8. Verify the Maven installation by running `mvn --version`.
- **Scala**:
    1. Go to the Scala website: [The Scala Programming Language (scala-lang.org)](https://www.scala-lang.org/).
    2. Scroll down and download the binary package for Scala 2.13.8 (e.g., scala-2.13.8.tgz).
    3. On the Hadoop server, create a `/scala` directory and download the Scala package using the command `curl -O https://downloads.lightbend.com/scala/2.13.8/scala-2.13.8.tgz`.
    4. Extract the downloaded package.
    5. Configure the Scala environment variables in the `~/.bash_profile` file:
```bash
# User specific environment and startup programs
PATH=$PATH:$HOME/bin
JAVA_HOME=/java
HADOOP_HOME=/hadoop
MAVEN_HOME=/maven
SCALA_HOME=/scala
PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$MAVEN_HOME/bin:$SCALA_HOME/bin
export JAVA_HOME
export HADOOP_HOME
export MAVEN_HOME
export SCALA_HOME
export PATH
```
    6. Use the command `source ~/.bash_profile` to make the environment variables take effect.
    7. Verify the Scala installation by running `scala --version`.

### 3.2 Compilation Process
1. Navigate to the Spark source code directory (e.g., `/spark`).
2. Set the Maven options in the `~/.bash_profile` file:
```bash
# User specific environment and startup programs
PATH=$PATH:$HOME/bin
JAVA_HOME=/java
HADOOP_HOME=/hadoop
MAVEN_HOME=/maven
PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$MAVEN_HOME/bin
MAVEN_OPTS="-Xss64m -Xmx2g -XX:ReservedCodeCacheSize=1g"
export JAVA_HOME
export HADOOP_HOME
export MAVEN_HOME
export MAVEN_OPTS
export PATH
```
3. Use the command `source ~/.bash_profile` to make the options take effect.
4. Review the `pom.xml` file in the Spark source code to understand the available profiles and properties.
5. Build Spark with Hive and JDBC support using the following Maven command (example with specific versions and profiles):
```bash
mvn -Phadoop-2.7 -Phive-2.3 -Pyarn -Phive-thriftserver -Pscala-2.13 -Djava.version=11 -Dmaven.version=3.8.4 -Dhadoop.version=2.9.2 -Dscala.version=2.13.8 -Dscala.binary.version=2.13 -DskipTests clean package
```
- The `-D` options set various properties such as Java version, Maven version, Hadoop version, and Scala version.
- The `-P` options enable specific profiles like Hadoop 2.7, Hive 2.3, YARN, Hive Thrift Server, and Scala 2.13.
6. If using Scala 2.13, you may need to switch the Scala version. Use the command `./dev/change-scala-version.sh 2.13` (make sure to be in the Spark root directory).
7. The compilation process may take some time as Maven downloads dependencies. If there are compilation errors, review the error messages and make necessary adjustments. For example, if there are issues related to Scala compatibility, ensure that the correct Scala version is used and all necessary dependencies are available.
8. If the compilation is successful, binary files will be generated in the `bin` directory of the Spark root directory. You can then start using Spark commands like `spark-shell`.

## 4. Compiling and Packaging a Distributable Binary
1. After successful compilation, you can package a distributable binary. In the Spark root directory, execute the following command (example with specific options):
```bash
./dev/make-distribution.sh --name 2.12 --tgz --mvn mvn -Phadoop-2.7 -Phive-2.3 -Pyarn -Phive-thriftserver -Pscala-2.12 -Djava.version=11 -Dmaven.version=3.8.4 -Dhadoop.version=2.9.2 -Dscala.version=2.12.15 -Dscala.binary-version=2.12 -DskipTests clean package
```
- The `--name` option specifies the name of the distribution.
- The `--tgz` option creates a compressed tarball.
- The `--mvn` option allows you to pass custom Maven commands.
2. The script will package the compiled files into a distributable format. After completion, a binary package (e.g., `spark-2.9.2-bin-2.12.tgz`) will be generated in the Spark root directory.

## 5. Single-Machine Startup
1. Distribute the compiled binary package (e.g., `spark-3.2.0-bin-2.12.tgz`) to the machine where you want to start Spark (e.g., Hadoop02). Use the command `scp spark-3.2.0-bin-2.12.tgz hadoop02:/spark/`.
2. On the target machine (e.g., Hadoop02), extract the package using `tar -zxvf spark-3.2.0-bin-2.12.tgz`.
3. Set the `SPARK_HOME` environment variable in the `~/.bash_profile` file and make it point to the Spark installation directory (e.g., `/spark`).
```bash
# User specific environment and startup programs
PATH=$PATH:$HOME/bin
JAVA_HOME=/java
HADOOP_HOME=/hadoop
SPARK_HOME=/spark
PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin
export JAVA_HOME
export SPARK_HOME
export HADOOP_HOME
export PATH
```
4. Use the command `source ~/.bash_profile` to make the environment variable take effect.
5. You can now start Spark in standalone mode using the `spark-shell` command. This will start a Spark session and provide a Scala REPL for interactive data analysis.

## 6. Cluster Deployment
1. Configure the Spark cluster settings in the `/conf/spark-env.sh` file. Here is an example configuration for a simple cluster with one master (Hadoop01) and two workers (Hadoop02 and Hadoop03):
```bash
# Options read when launching programs locally with
#./bin/run-example or./bin/spark-submit
# - HADOOP_CONF_DIR, to point Spark towards Hadoop configuration files
# - SPARK_LOCAL_IP, to set the IP address Spark binds to on this node
# - SPARK_PUBLIC_DNS, to set the public dns name of the driver program

# hadoop 的配置信息
HADOOP_CONF_DIR=/hadoop/etc/hadoop

# Options read by executors and drivers running inside the cluster
# - SPARK_LOCAL_IP, to set the IP address Spark binds to on this node
# - SPARK_PUBLIC_DNS, to set the public DNS name of the driver program
# - SPARK_LOCAL_DIRS, storage directories to use on this node for shuffle and RDD data
# - MESOS_NATIVE_JAVA_LIBRARY, to point to your libmesos.so if you use Mesos

# spark shuffle的数据目录，
SPARK_LOCAL_DIRS=/spark/shuffle_data

# Options read in YARN client/cluster mode
# - SPARK_CONF_DIR, Alternate conf dir. (Default: ${SPARK_HOME}/conf)
# - HADOOP_CONF_DIR, to point Spark towards Hadoop configuration files
# - YARN_CONF_DIR, to point Spark towards YARN configuration files when you use YARN
# - SPARK_EXECUTOR_CORES, Number of cores for the executors (Default: 1).
# - SPARK_EXECUTOR_MEMORY, Memory per Executor (e.g. 1000M, 2G) (Default: 1G)
# - SPARK_DRIVER_MEMORY, Memory for Driver (e.g. 1000M, 2G) (Default: 1G)

# yarn的配置文件在hadoop的配置文件中
YARN_CONF_DIR=$HADOOP_CONF_DIR
# 每个节点启动几个执行器
SPARK_EXECUTOR_CORES=1
# 每个执行器可以使用多大内存
SPARK_EXECUTOR_MEMORY=1800M
# 每个driver可以使用多大内存
SPARK_DRIVER_MEMORY=1800M

# Options for the daemons used in the standalone deploy mode
# - SPARK_MASTER_HOST, to bind the master to a different IP address or hostname
# - SPARK_MASTER_PORT / SPARK_MASTER_WEBUI_PORT, to use non-default ports for the master
# - SPARK_MASTER_OPTS, to set config properties only for the master (e.g. "-Dx=y")
# - SPARK_WORKER_CORES, to set the number of cores to use on this machine
# - SPARK_WORKER_MEMORY, to set how much total memory workers have to give executors (e.g. 1000m, 2g)
# - SPARK_WORKER_PORT / SPARK_WORKER_WEBUI_PORT, to use non-default ports for the worker
# - SPARK_WORKER_DIR, to set the working directory of worker processes
# - SPARK_WORKER_OPTS, to set config properties only for the worker (e.g. "-Dx=y")
# - SPARK_DAEMON_MEMORY, to allocate to the master, worker and history server themselves (default: 1g).
# - SPARK_HISTORY_OPTS, to set config properties only for the history server (e.g. "-Dx=y")
# - SPARK_SHUFFLE_OPTS, to set config properties only for the external shuffle service (e.g. "-Dx=y")
# - SPARK_DAEMON_JAVA_OPTS, to set config properties for all daemons (e.g. "-Dx=y")
# - SPARK_DAEMON_CLASSPATH, to set the classpath for all daemons
# - SPARK_PUBLIC_DNS, to set the public dns name of the master or workers

# 主节点
SPARK_MASTER_HOST=hadoop01
# 端口，不要和worker的端口重复，通信端口
SPARK_MASTER_PORT=4040
# 界面端口
SPARK_MASTER_WEBUI_PORT=8089
# 每个从节点启动的执行器
SPARK_WORKER_CORES=1
# 每个从节点可使用最大内存
SPARK_WORKER_MEMORY=1800M
# 从节点的通信端口
SPARK_WORKER_PORT=4040
# 界面端口
SPARK_WORKER_WEBUI_PORT=8089
# 从节点工作目录
SPARK_WORKER_DIR=/spark/worker

# Options for launcher
# - SPARK_LAUNCHER_OPTS, to set config properties and Java options for the launcher (e.g. "-Dx=y")

# Generic options for the daemons used in the standalone deploy mode
# - SPARK_CONF_DIR      Alternate conf dir. (Default: ${SPARK_HOME}/conf)
# - SPARK_LOG_DIR       Where log files are stored.  (Default: ${SPARK_HOME}/logs)
# - SPARK_LOG_MAX_FILES Max log files of Spark daemons can rotate to. Default is 5.
# - SPARK_PID_DIR       Where the pid file is stored. (Default: /tmp)
# - SPARK_IDENT_STRING  A string representing this instance of spark. (Default: $USER)
# - SPARK_NICENESS      The scheduling priority for daemons. (Default: 0)
# - SPARK_NO_DAEMONIZE  Run the proposed command in the foreground. It will not output a PID file.
# Options for native BLAS, like Intel MKL, OpenBLAS, and so on.
# You might get better performance to enable these options if using native BLAS (see SPARK-21305).
# - MKL_NUM_THREADS=1        Disable multi-threading of Intel MKL
# - OPENBLAS_NUM_THREADS=1   Disable multi-threading of OpenBLAS

# spark 的日志目录
SPARK_LOG_DIR=/spark/logs
```
2. Configure the list of worker nodes in the `/conf/workers` file. For example:
```
hadoop02
hadoop03
```
3. Distribute the configured Spark directory (including the `conf` directory with the above configurations) to all nodes in the cluster. You can use the `scp -r` command to copy the directory.
4. On the master node (e.g., Hadoop01), navigate to the `/spark/sbin` directory.
5. Start the Spark cluster using the following scripts:
    - Start the master: `./start-master.sh`
    - Start the workers: `./start-workers.sh`
    - Alternatively, you can use `./start-all.sh` to start both the master and the workers.
6. Verify the cluster status by accessing the Spark master web UI (e.g., `http://hadoop01:8089`). You should see the master and the registered worker nodes.

## 7. IDEA Integration
1. Install the Scala plugin in IDEA. Go to `Settings` -> `Plugins` and search for "Scala". Install the plugin and restart IDEA.
2. Configure the Scala SDK in IDEA.
    - Download the Scala binaries (if not already installed) and extract them to a local directory.
    - In IDEA, go to `File` -> `Project Structure` -> `SDKs`. Click the `+` button and select "Scala SDK". Point to the directory where Scala is installed.
3. Install sbt, the Scala build tool.
    - Go to the sbt website: [sbt - The interactive build tool (scala-sbt.org)](https://www.scala-sbt.org/).
    - Download the appropriate sbt package (e.g., sbt-1.6.1.zip).
    - Extract the downloaded package to a local directory and add the sbt binary directory to the system path.
4. Create a new Spark project in IDEA.
    - Select "sbt-based Scala project (recommended)" when creating a new project.
    - Configure the project settings such as project name, location, JDK, sbt version, Scala version, and package prefix.
5. Configure the project's `build.sbt` file. Here is an example configuration:



## Running Tests

Testing first requires [building Spark](#building-spark). Once Spark is built, tests
can be run using:

    ./dev/run-tests

Please see the guidance on how to
[run tests for a module, or individual tests](http://spark.apache.org/developer-tools.html#individual-tests).




# WHLB Spark Algorithm Related Class

This README provides an overview and usage instructions for the Spark algorithms implemented based on the "Spark Environment Load Balancing Optimization Strategy Research" paper. The algorithms aim to address data skew issues in Spark applications and improve overall performance.

## 1. Introduction
In Spark applications, data skew during the Shuffle stage can lead to inefficient resource utilization and increased processing time. The algorithms presented here implement the strategies proposed in the research paper to mitigate data skew and optimize load balancing.

## 2. Algorithms

### 2.1 Sampling Method
The sampling method uses an improved reservoir sampling algorithm to obtain a representative sample of the data. This sample is used to estimate data distribution characteristics, including total data size, number of data items, and key-value statistics.

#### Usage
```scala
import scala.collection.mutable.ListBuffer
import scala.util.Random

object Sampling {

  // Improved reservoir sampling algorithm
  def improvedSampling(rdd: RDD[(String, Int)], sampleRate: Double, partitionNum: Int, resampleThreshold: Int): (ListBuffer[(String, Int)], Long, Long, Map[String, (Int, Double)]) = {
    val totalSampleNum = (rdd.count() * sampleRate).toInt
    val samplesPerPartition = totalSampleNum / partitionNum

    var totalDataSize = 0L
    var totalDataItems = 0L
    val sampledData = ListBuffer[(String, Int)]()
    val keyStats = collection.mutable.Map[String, (Int, Double)]()

    val rddWithIndex = rdd.zipWithIndex().map { case ((key, value), index) => ((index / samplesPerPartition).toInt, (key, value)) }
    val partitionedRDD = rddWithIndex.partitionBy(new HashPartitioner(partitionNum))

    partitionedRDD.foreachPartition { partition =>
      var partitionDataItems = 0
      var partitionDataSize = 0L
      val localSampledData = ListBuffer[(String, Int)]()
      val localKeyStats = collection.mutable.Map[String, (Int, Double)]()

      partition.foreach { case (_, (key, value)) =>
        partitionDataItems += 1
        partitionDataSize += value
        totalDataItems += 1
        totalDataSize += value

        if (partitionDataItems <= samplesPerPartition) {
          localSampledData += ((key, value))
        } else {
          val randomValue = Random.nextInt(partitionDataItems)
          if (randomValue < samplesPerPartition) {
            localSampledData(randomValue) = ((key, value))
          }
        }

        val keyStat = localKeyStats.getOrElse(key, (0, 0.0))
        localKeyStats(key) = (keyStat._1 + 1, keyStat._2 + value)
      }

      // Check if resampling is needed
      if (partitionDataItems > resampleThreshold * samplesPerPartition) {
        val resampleSize = resampleThreshold * samplesPerPartition
        val resampledPartition = partition.take(resampleSize)
        resampledPartition.foreach { case (_, (key, value)) =>
          localSampledData += ((key, value))
          val keyStat = localKeyStats.getOrElse(key, (0, 0.0))
          localKeyStats(key) = (keyStat._1 + 1, keyStat._2 + value)
        }
      }

      sampledData ++= localSampledData
      localKeyStats.foreach { case (key, (count, totalSize)) =>
        keyStats(key) = (count, totalSize / count)
      }
    }

    val finalSampleRate = sampledData.size.toDouble / totalDataItems
    (sampledData, totalDataItems, totalDataSize, keyStats.toMap)
  }
}
```

### 2.2 Parallelism Adjustment Algorithm
This algorithm adjusts the parallelism of Spark stages based on the sampled data and available resources. It calculates an optimal number of partitions to balance the workload and improve resource utilization.

#### Usage
```scala
object ParallelismAdjustment {

  def adjustParallelism(sampledData: ListBuffer[(String, Int)], totalDataItems: Long, totalDataSize: Long, executorCores: Int, executorMemory: Long, executorNum: Int, sampleRate: Double): Int = {
    val x = 1.5 // Assume data size to memory demand ratio, adjust according to actual situation
    val cd = (x * totalDataSize) / (executorCores * executorNum)
    val mmax = executorMemory / executorCores
    val mmin = mmax / 2

    val np1 = ((cd / mmax).toInt, (cd / mmin).toInt)

    val keySizes = sampledData.groupBy(_._1).mapValues(_.map(_._2).sum)
    val keyFrequencies = keySizes.mapValues(_.toDouble / totalDataItems)
    val largeTaskSizes = keySizes.filter(_._2 > cd).values
    val avgLargeTaskSize = if (largeTaskSizes.nonEmpty) {
      val weightedSum = largeTaskSizes.zip(keyFrequencies.values).map { case (size, freq) => size * freq }.sum
      weightedSum / keyFrequencies.values.sum
    } else {
      cd
    }

    val np2 = ((x * avgLargeTaskSize / mmax).toInt, (x * avgLargeTaskSize / mmin).toInt)

    val npRange = (math.max(np1._1, np2._1), math.min(np1._2, np2._2))
    val finalBatches = (npRange._1 + npRange._2) / 2
    val parallelism = math.max(1, finalBatches).toInt * executorCores * executorNum

    parallelism
  }
}
```

### 2.3 Weighted Hash Repartitioning Algorithm
This algorithm re-partitions the data based on weighted hashing. It considers the average tuple size and frequency of keys to distribute the data more evenly across partitions.

#### Usage
```scala
import com.google.common.hash.Hashing

object WeightedHashRepartition {

  def weightedHashRepartition(sampledData: ListBuffer[(String, Int)], expectedPartitionNum: Int): Map[String, Int] = {
    val keyStats = sampledData.groupBy(_._1).mapValues(values => (values.length, values.map(_._2).sum / values.length))
    val totalWeight = keyStats.values.map { case (count, avgSize) => count * avgSize }.sum
    val avgWeightPerPartition = totalWeight / expectedPartitionNum

    val partitionHeap = new mutable.PriorityQueue[(Int, Double)](Ordering.by[(Int, Double), Double](_._2))(expectedPartitionNum)
    (0 until expectedPartitionNum).foreach { i =>
      partitionHeap += ((i, avgWeightPerPartition))
    }

    val keyToPartition = mutable.Map[String, Int]()
    keyStats.foreach { case (key, (count, avgSize)) =>
      val weight = count * avgSize
      val hashValue = Hashing.murmur3_32().hashString(key).asInt()
      val tempPartitionId = hashValue % expectedPartitionNum
      val currentPartitionWeight = partitionHeap(tempPartitionId)._2

      if (currentPartitionWeight >= weight) {
        partitionHeap(tempPartitionId) = (tempPartitionId, currentPartitionWeight - weight)
        keyToPartition(key) = tempPartitionId
      } else {
        val minWeightPartition = partitionHeap.dequeue()
        partitionHeap += (minWeightPartition._1, minWeightPartition._2 - weight)
        keyToPartition(key) = minWeightPartition._1
      }
    }

    keyToPartition.toMap
  }
}
```

### 2.4 Data Scheduling Algorithm
The data scheduling algorithm assigns tasks to nodes based on the Longest Processing Time (LPT) strategy. It aims to balance the workload across nodes and minimize the overall processing time.

#### Usage
```scala
import scala.collection.mutable.TreeMap

object DataScheduling {

  def scheduleTasks(taskSizes: Map[String, Long], nodeInfos: Seq[String]): Map[String, String] = {
    val taskSizeMap = new TreeMap[String, Long]()(Ordering.by(-_._2))
    taskSizes.foreach(taskSizeMap.put)

    val nodeLoads = nodeInfos.map(node => (node, 0L)).toMap
    val nodeLoadQueue = new mutable.PriorityQueue[(String, Long)](Ordering.by[(String, Long), Long](_._2))(nodeInfos.length)
    nodeLoads.foreach { case (node, load) =>
      nodeLoadQueue += ((node, load))
    }

    val taskAssignments = mutable.Map[String, String]()
    while (taskSizeMap.nonEmpty) {
      val (taskId, taskSize) = taskSizeMap.firstKey -> taskSizeMap.firstKey
      taskSizeMap.remove(taskId)

      val (minLoadedNode, _) = nodeLoadQueue.dequeue()
      taskAssignments(taskId) = minLoadedNode
      nodeLoadQueue += (minLoadedNode, nodeLoads(minLoadedNode) + taskSize)
    }

    taskAssignments.toMap
  }
}
```

## 3. Example Usage
Here is an example of how to use these algorithms in a Spark application:

```scala
import org.apache.spark.{SparkConf, SparkContext}

object WHLBTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("WHLBExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Assume there is an RDD[(String, Int)] dataset
    val datasetRDD = sc.parallelize(List(("key1", 1), ("key2", 2), ("key1", 3), ("key3", 4), ("key2", 5), ("key4", 6)))

    // Set sampling rate, partition number, and resample threshold
    val sampleRate = 0.5
    val partitionNum = 2
    val resampleThreshold = 4

    // Perform sampling
    val (sampledData, totalDataItems, totalDataSize, keyStats) = Sampling.improvedSampling(datasetRDD, sampleRate, partitionNum, resampleThreshold)

    // Adjust parallelism
    val executorCores = 2
    val executorMemory = 4096L
    val executorNum = 2
    val parallelism = ParallelismAdjustment.adjustParallelism(sampledData, totalDataItems, totalDataSize, executorCores, executorMemory, executorNum, sampleRate)

    // Repartition based on weighted hash
    val expectedPartitionNum = 3
    val keyToPartition = WeightedHashRepartition.weightedHashRepartition(sampledData, expectedPartitionNum)

    // Simulate task sizes and node information
    val taskSizes = keyToPartition.map { case (key, partitionId) => (s"task_$key", keyStats(key)._2.toLong) }
    val nodeInfos = Seq("node1", "node2")

    // Perform data scheduling
    val taskAssignments = DataScheduling.scheduleTasks(taskSizes, nodeInfos)

    taskAssignments.foreach(println)

    sc.stop()
  }
}
```

## 4. Notes
- The parameter values used in the example (such as `x`, `sampleRate`, `resampleThreshold`, etc.) should be adjusted according to the characteristics of the actual dataset and the Spark cluster configuration.
- The key-value pair type in the code is `(String, Int)`. You may need to modify it according to your actual data type.




# Hibench
The Hibench source code is in the root directory, ready to be compiled and used as needed



# DataGen
python was used to generate the data. Due to the large size of the relevant data, it exceeded the maximum size limit of github for a single file. If you have relevant data requirements, you can contact me
