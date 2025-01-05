package org.apache.spark.rdd

import scala.collection.mutable.ListBuffer
import scala.util.Random

object Sampling {

  // 改进的水塘抽样算法
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

      // 检查是否需要重采样
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