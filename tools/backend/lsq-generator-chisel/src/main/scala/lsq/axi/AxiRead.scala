//===- AxiRead.scala ------------------------------------------*- Scala -*-===//
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

class AxiRead(config: LsqConfigs) extends Module {
  override def desiredName: String = "AXI_READ"

  val io = IO(new Bundle {

    val loadDataFromMem = Output(UInt(config.dataWidth.W))
    val loadAddrToMem = Input(UInt(config.addrWidth.W))
    val loadQIdxForDataOut = Output(UInt(config.dataWidth.W))
    val loadQIdxForDataOutValid = Output(Bool())
    val loadQIdxForAddrIn = Flipped(Decoupled(UInt(config.addrWidth.W)))

    // read address channel
    val ARID = Output(UInt(config.bufferIdxWidth))
    val ARADDR = Output(UInt(config.addrWidth.W))
    val ARLEN = Output(UInt(8.W))
    val ARSIZE = Output(UInt(3.W))
    val ARBURST = Output(UInt(2.W))
    val ARLOCK = Output(Bool())
    val ARCACHE = Output(UInt(4.W))
    val ARPROT = Output(UInt(3.W))
    val ARQOS = Output(UInt(4.W))
    val ARREGION = Output(UInt(4.W))
    val ARVALID = Output(Bool())
    val ARREADY = Input(Bool())

    // read data channel
    val RID = Input(UInt(config.bufferIdxWidth))
    val RDATA = Input(UInt(config.dataWidth.W))
    val RRESP = Input(UInt(2.W))
    val RLAST = Input(Bool())
    val RVALID = Input(Bool())
    val RREADY = Output(Bool())

  })

  io.ARLEN := 0.U
  io.ARSIZE := log2Ceil(config.dataWidth).U
  io.ARBURST := 1.U
  io.ARLOCK := false.B
  io.ARCACHE := 0.U
  io.ARPROT := 0.U
  io.ARQOS := 0.U
  io.ARREGION := 0.U

  io.ARADDR := io.loadAddrToMem
  io.loadDataFromMem := io.RDATA
  io.RREADY := true.B

  val idxArray = RegInit(
    VecInit(Seq.fill(config.bufferDepth)(0.U(config.fifoIdxWidth)))
  )
  val waitingForData = RegInit(VecInit(Seq.fill(config.bufferDepth)(false.B)))

  val firstFreeIdx = PriorityEncoder(VecInit(waitingForData map (x => !x)))

  val hasFreeIdx = waitingForData.exists((x: Bool) => !x)

  io.ARVALID := hasFreeIdx && io.loadQIdxForAddrIn.valid
  io.ARADDR := io.loadAddrToMem
  io.ARID := firstFreeIdx
  io.loadQIdxForAddrIn.ready := hasFreeIdx && io.ARREADY

  for (i <- 0 until config.bufferDepth) {
    when(
      i.U === firstFreeIdx && hasFreeIdx && io.ARREADY && io.loadQIdxForAddrIn.valid
    ) {
      idxArray(i) := io.loadQIdxForAddrIn.bits
      waitingForData(i) := true.B
    }.elsewhen(i.U === io.RID && io.RVALID && io.RRESP === 0.U) {
      waitingForData(i) := false.B
    }
  }

  io.loadQIdxForDataOut := idxArray(io.RID)
  io.loadQIdxForDataOutValid := (io.RVALID && io.RRESP === 0.U)
  io.loadDataFromMem := io.RDATA

}
