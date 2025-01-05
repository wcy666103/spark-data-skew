package org.apache.spark.rdd

import scala.collection.mutable.ListBuffer

object ParallelismAdjustment {

  def adjustParallelism(sampledData: ListBuffer[(String, Int)], totalDataItems: Long, totalDataSize: Long, executorCores: Int, executorMemory: Long, executorNum: Int, sampleRate: Double): Int = {
    val x = 1.5 // 假设数据量与内存需求比值为1.5，可根据实际情况调整
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