//===- StoreDataPort.scala ------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq.port

import chisel3._
import chisel3.util._
import lsq.config.LsqConfigs

class StoreDataPort(config: LsqConfigs) extends Module {
  override def desiredName: String = "STORE_DATA_PORT_" + config.name

  val io = IO(new Bundle {
    // interface to previous
    val dataFromPrev = Flipped(Decoupled(UInt(config.dataWidth.W)))
    // interface to GA
    val portEnable = Input(Bool())

    // interface to LQ
    val storeDataEnable = Output(Bool())
    val dataToStoreQueue = Output(UInt(config.dataWidth.W))
  })

  val cnt = RegInit(0.U(log2Ceil(config.fifoDepth_S + 1).W))

  // updating counter
  when(io.portEnable && !io.storeDataEnable && cnt =/= config.fifoDepth_S.U) {
    cnt := cnt + 1.U
  }.elsewhen(io.storeDataEnable && !io.portEnable && cnt =/= 0.U) {
    cnt := cnt - 1.U
  }

  io.dataToStoreQueue := io.dataFromPrev.bits
  io.storeDataEnable := cnt > 0.U && io.dataFromPrev.valid
  io.dataFromPrev.ready := cnt > 0.U
}
