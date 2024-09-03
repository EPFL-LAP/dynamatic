//===- LSQSlave.scala -----------------------------------------*- Scala -*-===//
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
import lsq.config.LsqConfigs
import lsq.port._
import lsq.queues._

import scala.math.{max, min}

class LSQSlave(lsqConfig: LsqConfigs) extends Module {
  override def desiredName: String = lsqConfig.name

  val io = IO(new Bundle {
    val ldAddrToMC = Decoupled(UInt(lsqConfig.addrWidth.W))
    val ldDataFromMC = Flipped(Decoupled(UInt(lsqConfig.addrWidth.W)))
    val stAddrToMC = Decoupled(UInt(lsqConfig.addrWidth.W))
    val stDataToMC = Decoupled(UInt(lsqConfig.dataWidth.W))

    // Control signals
    val ctrl = Vec(lsqConfig.numBBs, Flipped(Decoupled(UInt(0.W))))

    // Load ports
    val ldAddr = Vec(
      lsqConfig.numLoadPorts,
      Flipped(Decoupled(UInt(lsqConfig.addrWidth.W)))
    ) 
    val ldData = Vec(
      lsqConfig.numLoadPorts,
      Decoupled(UInt(lsqConfig.dataWidth.W))
    ) 

    // Store ports
    val stAddr = Vec(
      lsqConfig.numStorePorts,
      Flipped(Decoupled(UInt(lsqConfig.addrWidth.W)))
    ) 
    val stData = Vec(
      lsqConfig.numStorePorts,
      Flipped(Decoupled(UInt(lsqConfig.dataWidth.W)))
    )
  })

  require(lsqConfig.fifoDepth_L > 1)
  require(lsqConfig.fifoDepth_S > 1)
  val bbStoreOffsets = Wire(
    Vec(lsqConfig.fifoDepth_S, UInt(log2Ceil(lsqConfig.fifoDepth_L).W))
  )
  val bbStorePorts = Wire(
    Vec(
      lsqConfig.fifoDepth_S,
      UInt(max(1, log2Ceil(lsqConfig.numStorePorts)).W)
    )
  )
  val bbNumStores = Wire(
    UInt(
      max(
        1,
        log2Ceil(min(lsqConfig.numStorePorts, lsqConfig.fifoDepth_S) + 1)
      ).W
    )
  )
  val storeTail = Wire(UInt(lsqConfig.fifoDepth_S.W))
  val storeHead = Wire(UInt(lsqConfig.fifoDepth_S.W))
  val storeEmpty = Wire(Bool())
  val bbLoadOffsets = Wire(
    Vec(lsqConfig.fifoDepth_L, UInt(log2Ceil(lsqConfig.fifoDepth_S).W))
  )
  val bbLoadPorts = Wire(
    Vec(lsqConfig.fifoDepth_L, UInt(max(1, log2Ceil(lsqConfig.numLoadPorts)).W))
  )
  val bbNumLoads = Wire(
    UInt(
      max(1, log2Ceil(min(lsqConfig.numLoadPorts, lsqConfig.fifoDepth_L) + 1)).W
    )
  )
  val loadTail = Wire(UInt(lsqConfig.fifoDepth_L.W))
  val loadHead = Wire(UInt(lsqConfig.fifoDepth_L.W))
  val loadEmpty = Wire(Bool())
  val loadPortsEnable = Wire(Vec(lsqConfig.numLoadPorts, Bool()))
  val storePortsEnable = Wire(Vec(lsqConfig.numStorePorts, Bool()))
  val bbStart = Wire(Bool())
  val storeAddressDone = Wire(Vec(lsqConfig.fifoDepth_S, Bool()))
  val storeDataDone = Wire(Vec(lsqConfig.fifoDepth_S, Bool()))
  val storeAddressQueue = Wire(
    Vec(lsqConfig.fifoDepth_S, UInt(lsqConfig.addrWidth.W))
  )
  val storeDataQueue = Wire(
    Vec(lsqConfig.fifoDepth_S, UInt(lsqConfig.dataWidth.W))
  )
  val loadAddressDone = Wire(Vec(lsqConfig.fifoDepth_L, Bool()))
  val loadDataDone = Wire(Vec(lsqConfig.fifoDepth_L, Bool()))
  val loadAddressQueue = Wire(
    Vec(lsqConfig.fifoDepth_L, UInt(lsqConfig.addrWidth.W))
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

  // component instantiation
  val storeQ = Module(new StoreQueue(lsqConfig))
  val loadQ = Module(new LoadQueue(lsqConfig))
  val GA = Module(new GroupAllocator(lsqConfig))

  val readPorts = VecInit(Seq.fill(lsqConfig.numLoadPorts) {
    Module(new LoadPort(lsqConfig)).io
  })
  val writeDataPorts = VecInit(Seq.fill(lsqConfig.numStorePorts) {
    Module(new StoreDataPort(lsqConfig)).io
  })
  val writeAddressPorts = VecInit(Seq.fill(lsqConfig.numStorePorts) {
    Module(new StoreAddrPort(lsqConfig)).io
  })

  // Load address to MC (ignore ready)
  io.ldAddrToMC.bits := loadQ.io.loadAddrToMem 
  io.ldAddrToMC.valid := loadQ.io.loadEnableToMem
  // Load data from MC (ignore valid)
  loadQ.io.loadDataFromMem := io.ldDataFromMC.bits
  io.ldDataFromMC.ready := 1.B

  // Store address to MC (ignore ready)
  io.stAddrToMC.bits := storeQ.io.storeAddrToMem
  io.stAddrToMC.valid := storeQ.io.storeEnableToMem  
  // Store data to MC (ignore ready)
  io.stDataToMC.bits := storeQ.io.storeDataToMem
  io.stDataToMC.valid := storeQ.io.storeEnableToMem
  
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
  
  io.ctrl.zip(GA.io.bbStartSignals).zip(GA.io.readyToPrevious).foreach{
    case ((groupStart, allocatorStart), allocatorReady) => {
      allocatorStart := groupStart.valid
      groupStart.ready := allocatorReady
      groupStart.bits := DontCare
    }
  }  

  for (i <- 0 until lsqConfig.numLoadPorts) {
    dataFromLoadQueue(i).valid := loadQ.io.loadPorts(i).valid
    dataFromLoadQueue(i).bits := loadQ.io.loadPorts(i).bits
    loadQ.io.loadPorts(i).ready := dataFromLoadQueue(i).ready

    loadQ.io.addrFromLoadPorts(i) := addressToLoadQueue(i)
    loadQ.io.loadAddrEnable(i) := loadAddressEnable(i)
  }

  loadQ.io.memIsReadyForLoads := 1.B

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
  
  storeQ.io.memIsReadyForStores := 1.B

  for (i <- 0 until lsqConfig.numLoadPorts) {
    readPorts(i).addrFromPrev <> io.ldAddr(i)
    readPorts(i).portEnable := loadPortsEnable(i)
    io.ldData(i) <> readPorts(i).dataToNext
    loadAddressEnable(i) := readPorts(i).loadAddrEnable
    addressToLoadQueue(i) := readPorts(i).addrToLoadQueue
    dataFromLoadQueue(i).ready := readPorts(i).dataFromLoadQueue.ready
    readPorts(i).dataFromLoadQueue.valid := dataFromLoadQueue(i).valid
    readPorts(i).dataFromLoadQueue.bits := dataFromLoadQueue(i).bits
  }

  for (i <- 0 until lsqConfig.numStorePorts) {
    writeDataPorts(i).dataFromPrev <> io.stData(i)
    writeDataPorts(i).portEnable := storePortsEnable(i)
    storeDataEnable(i) := writeDataPorts(i).storeDataEnable
    dataToStoreQueue(i) := writeDataPorts(i).dataToStoreQueue

    writeAddressPorts(i).addrFromPrev <> io.stAddr(i)
    writeAddressPorts(i).portEnable := storePortsEnable(i)
    storeAddressEnable(i) := writeAddressPorts(i).storeAddrEnable
    addressToStoreQueue(i) := writeAddressPorts(i).addrToStoreQueue
  }

}
