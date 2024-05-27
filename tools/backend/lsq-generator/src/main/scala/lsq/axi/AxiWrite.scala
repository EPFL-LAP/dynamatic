//===- AxiWrite.scala -----------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq.axi

import chisel3._
import chisel3.util._
import lsq.config.LsqConfigs

class AxiWrite(config: LsqConfigs) extends Module {
  override def desiredName: String = "AXI_WRITE"

  val io = IO(new Bundle {

    val storeAddrToMem = Input(UInt(config.addrWidth.W))
    val storeDataToMem = Input(UInt(config.dataWidth.W))
    val storeQIdxInToAW = Flipped(Decoupled(UInt(config.fifoIdxWidth)))
    val storeQIdxOutFromAW = Output(UInt(config.fifoIdxWidth))
    val storeQIdxOutFromAWValid = Output(Bool())

    // write address channel
    val AWID = Output(UInt(config.bufferIdxWidth))
    val AWADDR = Output(UInt(config.addrWidth.W))
    val AWLEN = Output(UInt(8.W))
    val AWSIZE = Output(UInt(3.W))
    val AWBURST = Output(UInt(2.W))
    val AWLOCK = Output(Bool())
    val AWCACHE = Output(UInt(4.W))
    val AWPROT = Output(UInt(3.W))
    val AWQOS = Output(UInt(4.W))
    val AWREGION = Output(UInt(4.W))
    val AWVALID = Output(Bool())
    val AWREADY = Input(Bool())

    // write data channel
    val WID = Output(UInt(config.bufferIdxWidth))
    val WDATA = Output(UInt(config.dataWidth.W))
    val WSTRB = Output(UInt((config.dataWidth / 8).W))
    val WLAST = Output(Bool())
    val WVALID = Output(Bool())
    val WREADY = Input(Bool())

    // write response channel
    val BID = Input(UInt(config.bufferIdxWidth))
    val BRESP = Input(UInt(2.W))
    val BVALID = Input(Bool())
    val BREADY = Output(Bool())

  })

  val idxArray = RegInit(
    VecInit(Seq.fill(config.bufferDepth)(0.U(config.fifoIdxWidth)))
  )
  val waitingForResponse = RegInit(
    VecInit(Seq.fill(config.bufferDepth)(false.B))
  )
  val firstFreeIdx = PriorityEncoder(VecInit(waitingForResponse map (x => !x)))
  val hasFreeIdx = waitingForResponse.exists((x: Bool) => !x)

  io.AWLEN := 0.U
  io.AWSIZE := log2Ceil(config.dataWidth).U
  io.AWBURST := 1.U
  io.AWLOCK := false.B
  io.AWCACHE := 0.U
  io.AWPROT := 0.U
  io.AWQOS := 0.U
  io.AWREGION := 0.U
  io.WSTRB := ~0.U

  io.storeQIdxOutFromAWValid := io.BVALID && (io.BRESP === 0.U)
  io.storeQIdxOutFromAW := idxArray(io.BID)

  val idle :: waitForFlags :: Nil = Enum(2)
  val state = RegInit(idle)

  io.AWADDR := io.storeAddrToMem
  io.WDATA := io.storeDataToMem
  io.BREADY := true.B

  val addrReady = RegInit(false.B)
  val dataReady = RegInit(false.B)
  val addrValid = RegInit(false.B)
  val dataValid = RegInit(false.B)

  io.AWVALID := addrValid || (io.storeQIdxInToAW.valid && state === idle)
  io.WVALID := dataValid || (io.storeQIdxInToAW.valid && state === idle)
  io.WLAST := dataValid || (io.storeQIdxInToAW.valid && state === idle)
  io.WID := firstFreeIdx
  io.AWID := firstFreeIdx
  io.storeQIdxInToAW.ready := false.B

  switch(state) {
    is(idle) {
      when(io.storeQIdxInToAW.valid && hasFreeIdx) {
        addrReady := io.AWREADY
        dataReady := io.WREADY
        when(io.AWREADY && io.WREADY) {
          state := idle
          io.storeQIdxInToAW.ready := true.B
        }.otherwise {
          state := waitForFlags
          when(!io.AWREADY) {
            addrValid := true.B
          }
          when(!io.WREADY) {
            dataValid := true.B
          }
        }
      }
    }

    is(waitForFlags) {
      when((addrReady || io.AWREADY) && (dataReady || io.WREADY)) {
        addrReady := false.B
        dataReady := false.B
        addrValid := false.B
        dataValid := false.B
        state := idle
        io.storeQIdxInToAW.ready := true.B
      }.elsewhen(io.AWREADY || io.WREADY) {
        when(io.AWREADY) {
          addrReady := true.B
          addrValid := false.B
        }
        when(io.WREADY) {
          dataReady := true.B
          dataValid := false.B
        }
      }
    }
  }

  for (i <- 0 until config.bufferDepth) {
    when(i.U === firstFreeIdx && io.storeQIdxInToAW.ready) {
      waitingForResponse(i) := true.B
      idxArray(firstFreeIdx) := io.storeQIdxInToAW.bits
    }.elsewhen(i.U === io.BID) {
      waitingForResponse(i) := false.B
    }
  }
}
