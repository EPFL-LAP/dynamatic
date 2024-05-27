//===- LoadPort.scala -----------------------------------------*- Scala -*-===//
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

class LoadPort(config: LsqConfigs) extends Module {
  override def desiredName: String = "LOAD_PORT_" + config.name

  val io = IO(new Bundle {
    // interface to previous
    val addrFromPrev = Flipped(Decoupled(UInt(config.addrWidth.W)))
    // interface to GA
    val portEnable = Input(Bool())
    // interface to next
    val dataToNext = Decoupled(UInt(config.dataWidth.W))
    // interface to LQ
    val loadAddrEnable = Output(Bool())
    val addrToLoadQueue = Output(UInt(config.addrWidth.W))
    val dataFromLoadQueue = Flipped(Decoupled(UInt(config.dataWidth.W)))
  })

  val cnt = RegInit(0.U(log2Ceil(config.fifoDepth_L + 1).W))

  // updating counter
  when(io.portEnable && !io.loadAddrEnable && cnt =/= config.fifoDepth_L.U) {
    cnt := cnt + 1.U
  }.elsewhen(io.loadAddrEnable && !io.portEnable && cnt =/= 0.U) {
    cnt := cnt - 1.U
  }

  io.addrToLoadQueue := io.addrFromPrev.bits
  io.loadAddrEnable := cnt > 0.U && io.addrFromPrev.valid
  io.addrFromPrev.ready := cnt > 0.U
  io.dataToNext <> io.dataFromLoadQueue
}
