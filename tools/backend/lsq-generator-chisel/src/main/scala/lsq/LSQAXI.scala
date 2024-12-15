//===- LSQAXI.scala -------------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq

import chisel3._
import chisel3.util._
import lsq.GA._
import lsq.axi._
import lsq.config.LsqConfigs
import lsq.port._
import lsq.queues._

import scala.math.{max, min}

class LSQAXI(lsqConfig: LsqConfigs) extends Module {
  override def desiredName: String = lsqConfig.name

  val io = IO(new Bundle {

    // read address channel
    val ARID = Output(UInt(lsqConfig.bufferIdxWidth))
    val ARADDR = Output(UInt(lsqConfig.addrWidth.W))
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
    val RID = Input(UInt(lsqConfig.bufferIdxWidth))
    val RDATA = Input(UInt(lsqConfig.dataWidth.W))
    val RRESP = Input(UInt(2.W))
    val RLAST = Input(Bool())
    val RVALID = Input(Bool())
    val RREADY = Output(Bool())

    // write address channel
    val AWID = Output(UInt(lsqConfig.bufferIdxWidth))
    val AWADDR = Output(UInt(lsqConfig.addrWidth.W))
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
    val WID = Output(UInt(lsqConfig.bufferIdxWidth))
    val WDATA = Output(UInt(lsqConfig.dataWidth.W))
    val WSTRB = Output(UInt((lsqConfig.dataWidth / 8).W))
    val WLAST = Output(Bool())
    val WVALID = Output(Bool())
    val WREADY = Input(Bool())

    // write response channel

    val BID = Input(UInt(lsqConfig.bufferIdxWidth))
    val BRESP = Input(UInt(2.W))
    val BVALID = Input(Bool())
    val BREADY = Output(Bool())

    ///////////////////////////////////////////////////////////////

    // previous interface related
    val bbStartSignals = Input(Vec(lsqConfig.numBBs, Bool()))
    val bbReadyToPrevious = Output(Vec(lsqConfig.numBBs, Bool()))
    // load ports related
    val previousAndLoadPorts = Vec(
      lsqConfig.numLoadPorts,
      Flipped(Decoupled(UInt(lsqConfig.addrWidth.W)))
    )
    val nextAndLoadPorts =
      Vec(lsqConfig.numLoadPorts, Decoupled(UInt(lsqConfig.dataWidth.W)))
    // store ports related
    val previousAndStoreAddressPorts = Vec(
      lsqConfig.numStorePorts,
      Flipped(Decoupled(UInt(lsqConfig.addrWidth.W)))
    )
    val previousAndStoreDataPorts = Vec(
      lsqConfig.numStorePorts,
      Flipped(Decoupled(UInt(lsqConfig.dataWidth.W)))
    )
    val queueIsEmpty = Output(Bool())
  })

  require(lsqConfig.fifoDepth > 1)
  val bbStoreOffsets = Wire(
    Vec(lsqConfig.fifoDepth, UInt(log2Ceil(lsqConfig.fifoDepth).W))
  )
  val bbStorePorts = Wire(
    Vec(lsqConfig.fifoDepth, UInt(max(1, log2Ceil(lsqConfig.numStorePorts)).W))
  )
  val bbNumStores = Wire(
    UInt(
      max(1, log2Ceil(min(lsqConfig.numStorePorts, lsqConfig.fifoDepth) + 1)).W
    )
  )
  val storeTail = Wire(UInt(lsqConfig.fifoDepth.W))
  val storeHead = Wire(UInt(lsqConfig.fifoDepth.W))
  val storeEmpty = Wire(Bool())
  val bbLoadOffsets = Wire(
    Vec(lsqConfig.fifoDepth, UInt(log2Ceil(lsqConfig.fifoDepth).W))
  )
  val bbLoadPorts = Wire(
    Vec(lsqConfig.fifoDepth, UInt(max(1, log2Ceil(lsqConfig.numLoadPorts)).W))
  )
  val bbNumLoads = Wire(
    UInt(
      max(1, log2Ceil(min(lsqConfig.numLoadPorts, lsqConfig.fifoDepth) + 1)).W
    )
  )
  val loadTail = Wire(UInt(lsqConfig.fifoDepth.W))
  val loadHead = Wire(UInt(lsqConfig.fifoDepth.W))
  val loadEmpty = Wire(Bool())
  val loadPortsEnable = Wire(Vec(lsqConfig.numLoadPorts, Bool()))
  val storePortsEnable = Wire(Vec(lsqConfig.numStorePorts, Bool()))
  val bbStart = Wire(Bool())
  val storeAddressDone = Wire(Vec(lsqConfig.fifoDepth, Bool()))
  val storeDataDone = Wire(Vec(lsqConfig.fifoDepth, Bool()))
  val storeAddressQueue = Wire(
    Vec(lsqConfig.fifoDepth, UInt(lsqConfig.addrWidth.W))
  )
  val storeDataQueue = Wire(
    Vec(lsqConfig.fifoDepth, UInt(lsqConfig.dataWidth.W))
  )
  val loadAddressDone = Wire(Vec(lsqConfig.fifoDepth, Bool()))
  val loadDataDone = Wire(Vec(lsqConfig.fifoDepth, Bool()))
  val loadAddressQueue = Wire(
    Vec(lsqConfig.fifoDepth, UInt(lsqConfig.addrWidth.W))
  )
  val dataFromLoadQueue = Wire(
    Vec(lsqConfig.numLoadPorts, Decoupled(UInt(lsqConfig.dataWidth.W)))
  )
  val loadAddressEnable = Wire(Vec(lsqConfig.numLoadPorts, Bool()))
  val addressToLoadQueue = Wire(
    Vec(lsqConfig.numLoadPorts, UInt(lsqConfig.addrWidth.W))
  )
  val dataToStoreQueue = Wire(
    Vec(lsqConfig.numStorePorts, UInt(lsqConfig.dataWidth.W))
  )
  val storeDataEnable = Wire(Vec(lsqConfig.numStorePorts, Bool()))
  val addressToStoreQueue = Wire(
    Vec(lsqConfig.numStorePorts, UInt(lsqConfig.addrWidth.W))
  )
  val storeAddressEnable = Wire(Vec(lsqConfig.numStorePorts, Bool()))

  //////////////////////////////////////////////////////////////////////////
  val storeDataToMem = Wire(UInt(lsqConfig.dataWidth.W))
  val storeAddressToMem = Wire(UInt(lsqConfig.addrWidth.W))
  val storeQIdxOut = Wire(Decoupled(UInt(lsqConfig.fifoIdxWidth)))
  val storeQIdxIn = Wire(UInt(lsqConfig.fifoIdxWidth))
  val storeQIdxInValid = Wire(Bool())

  //
  val loadDataFromMem = Wire(UInt(lsqConfig.dataWidth.W))
  val loadAddressToMem = Wire(UInt(lsqConfig.addrWidth.W))
  val loadQIdxForDataIn = Wire(UInt(lsqConfig.dataWidth.W))
  val loadQIdxForDataInValid = Wire(Bool())
  val loadQIdxForAddrOut = Wire(Decoupled(UInt(lsqConfig.addrWidth.W)))

  // component instantiation
  val storeQ = Module(new StoreQueueAxi(lsqConfig))
  val loadQ = Module(new LoadQueueAxi(lsqConfig))
  val GA = Module(new GroupAllocator(lsqConfig))
  val AXIWr = Module(new AxiWrite(lsqConfig))
  val AXIRd = Module(new AxiRead(lsqConfig))

  val lPorts = VecInit(Seq.fill(lsqConfig.numLoadPorts) {
    Module(new LoadPort(lsqConfig)).io
  })

  val sDataPorts = VecInit(Seq.fill(lsqConfig.numStorePorts) {
    Module(new StoreDataPort(lsqConfig)).io
  })

  val sAddressPorts = VecInit(Seq.fill(lsqConfig.numStorePorts) {
    Module(new StoreAddrPort(lsqConfig)).io
  })

  io.queueIsEmpty := storeEmpty && loadEmpty

  // Group Allocator assignments
  bbLoadOffsets := GA.io.bbLoadOffsets
  bbLoadPorts := GA.io.bbLoadPorts
  bbNumLoads := GA.io.bbNumLoads
  GA.io.loadTail := loadTail
  GA.io.loadHead := loadHead
  GA.io.loadEmpty := loadEmpty
  bbStoreOffsets := GA.io.bbStoreOffsets
  bbStorePorts := GA.io.bbStorePorts
  bbNumStores := GA.io.bbNumStores
  GA.io.storeTail := storeTail
  GA.io.storeHead := storeHead
  GA.io.storeEmpty := storeEmpty
  bbStart := GA.io.bbStart
  GA.io.bbStartSignals := io.bbStartSignals
  io.bbReadyToPrevious := GA.io.readyToPrevious
  loadPortsEnable := GA.io.loadPortsEnable
  storePortsEnable := GA.io.storePortsEnable

  // load queue assignments
  loadQ.io.storeTail := storeTail
  loadQ.io.storeHead := storeHead
  loadQ.io.storeEmpty := storeEmpty
  loadQ.io.storeAddrDone := storeAddressDone
  loadQ.io.storeDataDone := storeDataDone
  loadQ.io.storeAddrQueue := storeAddressQueue
  loadQ.io.storeDataQueue := storeDataQueue
  loadQ.io.bbStart := bbStart
  loadQ.io.bbLoadOffsets := bbLoadOffsets
  loadQ.io.bbLoadPorts := bbLoadPorts
  loadQ.io.bbNumLoads := bbNumLoads
  loadTail := loadQ.io.loadTail
  loadHead := loadQ.io.loadHead
  loadEmpty := loadQ.io.loadEmpty
  loadAddressDone := loadQ.io.loadAddrDone
  loadDataDone := loadQ.io.loadDataDone
  loadAddressQueue := loadQ.io.loadAddrQueue

  for (i <- 0 until lsqConfig.numLoadPorts) {
    loadQ.io.loadPorts(i) <> dataFromLoadQueue(i)
    loadQ.io.addrFromLoadPorts(i) := addressToLoadQueue(i)
    loadQ.io.loadAddrEnable(i) := loadAddressEnable(i)
  }

  loadQ.io.loadDataFromMem := loadDataFromMem
  loadAddressToMem := loadQ.io.loadAddrToMem
  loadQ.io.loadQIdxForDataIn := loadQIdxForDataIn
  loadQ.io.loadQIdxForDataInValid := loadQIdxForDataInValid
  loadQIdxForAddrOut.bits := loadQ.io.loadQIdxForAddrOut.bits
  loadQIdxForAddrOut.valid := loadQ.io.loadQIdxForAddrOut.valid
  loadQ.io.loadQIdxForAddrOut.ready := loadQIdxForAddrOut.ready

  // store queue assignments
  storeQ.io.loadTail := loadTail
  storeQ.io.loadHead := loadHead
  storeQ.io.loadEmpty := loadEmpty
  storeQ.io.loadAddressDone := loadAddressDone
  storeQ.io.loadDataDone := loadDataDone
  storeQ.io.loadAddressQueue := loadAddressQueue
  storeQ.io.bbStart := bbStart
  storeQ.io.bbStoreOffsets := bbStoreOffsets
  storeQ.io.bbStorePorts := bbStorePorts
  storeQ.io.bbNumStores := bbNumStores
  storeTail := storeQ.io.storeTail
  storeHead := storeQ.io.storeHead
  storeEmpty := storeQ.io.storeEmpty
  storeAddressDone := storeQ.io.storeAddrDone
  storeDataDone := storeQ.io.storeDataDone
  storeAddressQueue := storeQ.io.storeAddrQueue
  storeDataQueue := storeQ.io.storeDataQueue
  storeQ.io.storeDataEnable := storeDataEnable
  storeQ.io.dataFromStorePorts := dataToStoreQueue
  storeQ.io.storeAddrEnable := storeAddressEnable
  storeQ.io.addressFromStorePorts := addressToStoreQueue

  storeAddressToMem := storeQ.io.storeAddrToMem
  storeDataToMem := storeQ.io.storeDataToMem
  storeQIdxOut.valid := storeQ.io.storeQIdxOut.valid
  storeQIdxOut.bits := storeQ.io.storeQIdxOut.bits
  storeQ.io.storeQIdxOut.ready := storeQIdxOut.ready
  storeQ.io.storeQIdxIn := storeQIdxIn
  storeQ.io.storeQIdxInValid := storeQIdxInValid

  // axiRead
  AXIRd.io.loadAddrToMem := loadAddressToMem
  loadDataFromMem := AXIRd.io.loadDataFromMem
  loadQIdxForDataIn := AXIRd.io.loadQIdxForDataOut
  loadQIdxForDataInValid := AXIRd.io.loadQIdxForDataOutValid
  AXIRd.io.loadQIdxForAddrIn.bits := loadQIdxForAddrOut.bits
  AXIRd.io.loadQIdxForAddrIn.valid := loadQIdxForAddrOut.valid
  loadQIdxForAddrOut.ready := AXIRd.io.loadQIdxForAddrIn.ready

  io.ARID := AXIRd.io.ARID
  io.ARADDR := AXIRd.io.ARADDR
  io.ARLEN := AXIRd.io.ARLEN
  io.ARSIZE := AXIRd.io.ARSIZE
  io.ARBURST := AXIRd.io.ARBURST
  io.ARLOCK := AXIRd.io.ARLOCK
  io.ARCACHE := AXIRd.io.ARCACHE
  io.ARPROT := AXIRd.io.ARPROT
  io.ARQOS := AXIRd.io.ARQOS
  io.ARREGION := AXIRd.io.ARREGION
  io.ARVALID := AXIRd.io.ARVALID
  AXIRd.io.ARREADY := io.ARREADY

  AXIRd.io.RID := io.RID
  AXIRd.io.RDATA := io.RDATA
  AXIRd.io.RRESP := io.RRESP
  AXIRd.io.RLAST := io.RLAST
  AXIRd.io.RVALID := io.RVALID
  io.RREADY := AXIRd.io.RREADY

  // AxiWrite
  AXIWr.io.storeAddrToMem := storeAddressToMem
  AXIWr.io.storeDataToMem := storeDataToMem
  AXIWr.io.storeQIdxInToAW.bits := storeQIdxOut.bits
  AXIWr.io.storeQIdxInToAW.valid := storeQIdxOut.valid
  storeQIdxOut.ready := AXIWr.io.storeQIdxInToAW.ready

  storeQIdxIn := AXIWr.io.storeQIdxOutFromAW
  storeQIdxInValid := AXIWr.io.storeQIdxOutFromAWValid

  io.AWID := AXIWr.io.AWID
  io.AWADDR := AXIWr.io.AWADDR
  io.AWLEN := AXIWr.io.AWLEN
  io.AWSIZE := AXIWr.io.AWSIZE
  io.AWBURST := AXIWr.io.AWBURST
  io.AWLOCK := AXIWr.io.AWLOCK
  io.AWCACHE := AXIWr.io.AWCACHE
  io.AWPROT := AXIWr.io.AWPROT
  io.AWQOS := AXIWr.io.AWQOS
  io.AWREGION := AXIWr.io.AWREGION
  io.AWVALID := AXIWr.io.AWVALID
  AXIWr.io.AWREADY := io.AWREADY

  io.WID := AXIWr.io.WID
  io.WDATA := AXIWr.io.WDATA
  io.WSTRB := AXIWr.io.WSTRB
  io.WLAST := AXIWr.io.WLAST
  io.WVALID := AXIWr.io.WVALID
  AXIWr.io.WREADY := io.WREADY

  AXIWr.io.BID := io.BID
  AXIWr.io.BRESP := io.BRESP
  AXIWr.io.BVALID := io.BVALID
  io.BREADY := AXIWr.io.BREADY

  for (i <- 0 until lsqConfig.numLoadPorts) {
    lPorts(i).addrFromPrev <> io.previousAndLoadPorts(i)
    lPorts(i).portEnable := loadPortsEnable(i)
    io.nextAndLoadPorts(i) <> lPorts(i).dataToNext
    loadAddressEnable(i) := lPorts(i).loadAddrEnable
    addressToLoadQueue(i) := lPorts(i).addrToLoadQueue
    dataFromLoadQueue(i).ready := lPorts(i).dataFromLoadQueue.ready
    lPorts(i).dataFromLoadQueue.valid := dataFromLoadQueue(i).valid
    lPorts(i).dataFromLoadQueue.bits := dataFromLoadQueue(i).bits
  }

  for (i <- 0 until lsqConfig.numStorePorts) {
    sDataPorts(i).dataFromPrev <> io.previousAndStoreDataPorts(i)
    sDataPorts(i).portEnable := storePortsEnable(i)
    storeDataEnable(i) := sDataPorts(i).storeDataEnable
    dataToStoreQueue(i) := sDataPorts(i).dataToStoreQueue

    sAddressPorts(i).addrFromPrev <> io.previousAndStoreAddressPorts(i)
    sAddressPorts(i).portEnable := storePortsEnable(i)
    storeAddressEnable(i) := sAddressPorts(i).storeAddrEnable
    addressToStoreQueue(i) := sAddressPorts(i).addrToStoreQueue
  }

}
