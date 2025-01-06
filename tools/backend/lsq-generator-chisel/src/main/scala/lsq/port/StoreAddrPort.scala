//===- StoreAddrPort.scala ------------------------------------*- Scala -*-===//
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

class StoreAddrPort(config: LsqConfigs) extends Module {
  override def desiredName: String = "STORE_ADDR_PORT_" + config.name

  val io = IO(new Bundle {
    // interface to previous
    val addrFromPrev = Flipped(Decoupled(UInt(config.dataWidth.W)))
    // interface to GA
    val portEnable = Input(Bool())

    // interface to LQ
    val storeAddrEnable = Output(Bool())
    val addrToStoreQueue = Output(UInt(config.dataWidth.W))
  })

  val cnt = RegInit(0.U(log2Ceil(config.fifoDepth_S + 1).W))

  // updating counter
  when(io.portEnable && !io.storeAddrEnable && cnt =/= config.fifoDepth_S.U) {
    cnt := cnt + 1.U
  }.elsewhen(io.storeAddrEnable && !io.portEnable && cnt =/= 0.U) {
    cnt := cnt - 1.U
  }

  io.addrToStoreQueue := io.addrFromPrev.bits
  io.storeAddrEnable := cnt > 0.U && io.addrFromPrev.valid
  io.addrFromPrev.ready := cnt > 0.U
}
