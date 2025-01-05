package org.apache.sparktest

import org.apache.spark.rdd.{ParallelismAdjustment, Sampling}
import org.apache.spark.scheduler.DataScheduling
import org.apache.spark.{SparkConf, SparkContext, WeightedHashRepartition}

object WHLBTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("WHLBExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // 假设这里有一个RDD[(String, Int)]类型的数据集
    val datasetRDD = sc.parallelize(List(("key1", 1), ("key2", 2), ("key1", 3), ("key3", 4), ("key2", 5), ("key4", 6)))

    // 设置抽样率、分区数和重采样阈值
    val sampleRate = 0.5
    val partitionNum = 2
    val resampleThreshold = 4

    // 进行抽样
    val (sampledData, totalDataItems, totalDataSize, keyStats) = Sampling.improvedSampling(datasetRDD, sampleRate, partitionNum, resampleThreshold)

    // 调整并行度
    val executorCores = 2
    val executorMemory = 4096L
    val executorNum = 2
    val parallelism = ParallelismAdjustment.adjustParallelism(sampledData, totalDataItems, totalDataSize, executorCores, executorMemory, executorNum, sampleRate)

    // 根据权重重分区
    val expectedPartitionNum = 3
    val keyToPartition = WeightedHashRepartition.weightedHashRepartition(sampledData, expectedPartitionNum)

    // 模拟任务大小和节点信息
    val taskSizes = keyToPartition.map { case (key, partitionId) => (s"task_$key", keyStats(key)._2.toLong) }
    val nodeInfos = Seq("node1", "node2")

    // 进行数据调度
    val taskAssignments = DataScheduling.scheduleTasks(taskSizes, nodeInfos)

    taskAssignments.foreach(println)

    sc.stop()
  }
}