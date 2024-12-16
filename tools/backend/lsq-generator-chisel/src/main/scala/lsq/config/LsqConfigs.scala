//===- LsqConfigs.scala ---------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq.config

import chisel3._
import chisel3.util._
import upickle.default._
import scala.math._

case class LsqConfigs(
    dataWidth: Int,
    addrWidth: Int,
    fifoDepth: Int,
    fifoDepth_L: Int,
    fifoDepth_S: Int,
    numLoadPorts: Int,
    numStorePorts: Int,
    numBBs: Int,
    loadOffsets: List[List[Int]],
    storeOffsets: List[List[Int]],
    loadPorts: List[List[Int]],
    storePorts: List[List[Int]],
    numLoads: List[Int],
    numStores: List[Int],
    bufferDepth: Int,
    name: String,
    master: Boolean
) {
  def fifoDepth_M: Int = max(fifoDepth_S, fifoDepth_L)
  def loadPortIdWidth: Width = max(1, log2Ceil(numLoadPorts)).W
  def storePortIdWidth: Width = max(1, log2Ceil(numStorePorts)).W
  def fifoIdxWidth: Width = log2Ceil(fifoDepth).W
  def fifoIdxWidth_L: Width = log2Ceil(fifoDepth_L).W
  def fifoIdxWidth_S: Width = log2Ceil(fifoDepth_S).W
  def fifoIdxWidth_MAX: Width = {
    if (fifoDepth_L >= fifoDepth_S) {
      log2Ceil(fifoDepth_L).W
    } else {
      log2Ceil(fifoDepth_S).W
    }
  }
  def bufferIdxWidth: Width = max(1, log2Ceil(bufferDepth)).W
}
