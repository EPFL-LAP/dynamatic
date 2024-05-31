//===- AxiStoreQueue.scala ------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq.queues

import chisel3._
import chisel3.util._
import lsq.config._
import lsq.util._

import scala.math.{max, min}

class StoreQueueAxi(config: LsqConfigs) extends Module {
  override def desiredName: String = "STORE_QUEUE"

  val io = IO(new Bundle {
    // From group allocator
    val bbStart = Input(Bool())
    val bbStoreOffsets = Input(Vec(config.fifoDepth, UInt(config.fifoIdxWidth)))
    val bbStorePorts =
      Input(Vec(config.fifoDepth, UInt(config.storePortIdWidth)))
    val bbNumStores = Input(
      UInt(max(1, log2Ceil(min(config.numLoadPorts, config.fifoDepth) + 1)).W)
    )
    // To group allocator
    val storeTail = Output(UInt(config.fifoIdxWidth))
    val storeHead = Output(UInt(config.fifoIdxWidth))
    val storeEmpty = Output(Bool())
    //  From load queue
    val loadTail = Input(UInt(config.fifoIdxWidth))
    val loadHead = Input(UInt(config.fifoIdxWidth))
    val loadEmpty = Input(Bool())
    val loadAddressDone = Input(Vec(config.fifoDepth, Bool()))
    val loadDataDone = Input(Vec(config.fifoDepth, Bool()))
    val loadAddressQueue =
      Input(Vec(config.fifoDepth, UInt(config.addrWidth.W)))
    // To load queue
    val storeAddrDone = Output(Vec(config.fifoDepth, Bool()))
    val storeDataDone = Output(Vec(config.fifoDepth, Bool()))
    val storeAddrQueue = Output(Vec(config.fifoDepth, UInt(config.addrWidth.W)))
    val storeDataQueue = Output(Vec(config.fifoDepth, UInt(config.dataWidth.W)))
    // From store data ports
    val storeDataEnable = Input(Vec(config.numStorePorts, Bool()))
    val dataFromStorePorts =
      Input(Vec(config.numStorePorts, UInt(config.dataWidth.W)))
    // From store addr ports
    val storeAddrEnable = Input(Vec(config.numStorePorts, Bool()))
    val addressFromStorePorts =
      Input(Vec(config.numStorePorts, UInt(config.addrWidth.W)))
    // To memory
    val storeAddrToMem = Output(UInt(config.addrWidth.W))
    val storeDataToMem = Output(UInt(config.dataWidth.W))
    val storeQIdxOut = Decoupled(UInt(config.fifoIdxWidth))
    val storeQIdxIn = Input(UInt(config.fifoIdxWidth))
    val storeQIdxInValid = Input(Bool())
  })

  require(config.fifoDepth > 1)

  val dummyHead = RegInit(0.U(config.fifoIdxWidth))
  val head = RegInit(0.U(config.fifoIdxWidth))
  val tail = RegInit(0.U(config.fifoIdxWidth))

  val offsetQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth)(0.U(config.fifoIdxWidth)))
  )
  val portQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth)(0.U(config.storePortIdWidth)))
  )
  val addrQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth)(0.U(config.addrWidth.W)))
  )
  val dataQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth)(0.U(config.dataWidth.W)))
  )
  val addrKnown = RegInit(VecInit(Seq.fill(config.fifoDepth)(false.B)))
  val dataKnown = RegInit(VecInit(Seq.fill(config.fifoDepth)(false.B)))
  val allocatedEntries = RegInit(VecInit(Seq.fill(config.fifoDepth)(false.B)))
  val storeCompleted = RegInit(VecInit(Seq.fill(config.fifoDepth)(false.B)))
  val checkBits = RegInit(VecInit(Seq.fill(config.fifoDepth)(false.B)))

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Allocating a new BB
    * -----------------------------------------------------------------------------------------------------------------
    */

  val initBits = VecInit(
    Range(0, config.fifoDepth) map (idx =>
      (subMod(idx.U, tail, config.fifoDepth.U) < io.bbNumStores) && io.bbStart
    )
  )

  allocatedEntries := VecInit(
    (allocatedEntries zip initBits) map (x => x._1 || x._2)
  )

  for (idx <- 0 until config.fifoDepth) {
    when(initBits(idx)) {
      offsetQ(idx) := io.bbStoreOffsets(subMod(idx.U, tail, config.fifoDepth.U))
      portQ(idx) := io.bbStorePorts(subMod(idx.U, tail, config.fifoDepth.U))
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Find Conflicts
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** Computes whether any store should be checked for conflicts (at clock) No
    * need to check an entry if all loads up to the last preceding load are
    * already executed I.e., load head has gone past the last preceding load
    */
  val previousLoadHead = RegNext(io.loadHead)
  for (storeEntryIdx <- 0 until config.fifoDepth) {
    when(initBits(storeEntryIdx)) {
      checkBits(storeEntryIdx) := !(io.loadEmpty && addMod(
        io.bbStoreOffsets(subMod(storeEntryIdx.U, tail, config.fifoDepth.U)),
        1.U,
        config.fifoDepth.U
      ) === io.loadTail)
    }.otherwise {
      when(io.loadEmpty) {
        checkBits(storeEntryIdx) := false.B
      }.elsewhen(
        previousLoadHead <= offsetQ(storeEntryIdx) && offsetQ(
          storeEntryIdx
        ) < io.loadHead
      ) {
        checkBits(storeEntryIdx) := false.B
      }.elsewhen(
        previousLoadHead > io.loadHead &&
          !(io.loadHead <= offsetQ(storeEntryIdx) && offsetQ(
            storeEntryIdx
          ) < previousLoadHead)
      ) {
        checkBits(storeEntryIdx) := false.B
      }
    }
  }

  /** Load queue entries and corresponding flags
    */
  val loadAddrQ = io.loadAddressQueue
  val loadAddrKnown = io.loadAddressDone
  val loadDataKnown = io.loadDataDone

  /** Entries in the load queue are valid if they are between the load head and
    * the load tail
    */
  val validEntriesInLoadQ: Vec[Bool] = VecInit(
    Range(0, config.fifoDepth) map (idx =>
      Mux(
        io.loadHead < io.loadTail,
        io.loadHead <= idx.U && idx.U < io.loadTail,
        !io.loadEmpty && !(io.loadTail <= idx.U && idx.U < io.loadHead)
      )
    )
  )

  /** Load entries should be checked only if they are between the load head and
    * the last preceding load (i.e., offsetQ(head))
    */
  val loadsToCheck: Vec[Bool] = VecInit(
    Range(0, config.fifoDepth) map (idx =>
      Mux(
        io.loadHead <= offsetQ(dummyHead),
        io.loadHead <= idx.U && idx.U <= offsetQ(dummyHead),
        !(offsetQ(dummyHead) < idx.U && idx.U < io.loadHead)
      )
    )
  )

  /** Only the following entries in the loadQ should be checked for conflicts
    */
  val entriesToCheck: Vec[Bool] = VecInit(
    (loadsToCheck zip validEntriesInLoadQ) map
      (x => x._1 && x._2 && checkBits(dummyHead))
  )

  /** No-conflict flags for each load entry
    */
  val noConflicts: Vec[Bool] = Wire(Vec(config.fifoDepth, Bool()))
  for (loadEntryIdx <- 0 until config.fifoDepth) {
    noConflicts(loadEntryIdx) := !entriesToCheck(loadEntryIdx) || loadDataKnown(
      loadEntryIdx
    ) ||
      (loadAddrKnown(loadEntryIdx) && addrQ(dummyHead) =/= loadAddrQ(
        loadEntryIdx
      ))
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Store Data
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** A store request can be generated for storeQ(head) if its address and data
    * are known and if there are no conflicts
    */
  val storeRequest: Bool =
    addrKnown(dummyHead) && dataKnown(dummyHead) && !storeCompleted(
      dummyHead
    ) && noConflicts.forall(x => x)
  io.storeQIdxOut.bits := dummyHead

  /** Updating store completed flags (at clock) The store completed flags are
    * cleared for an entry when it is first allocated They are set when a store
    * request is generated for it.
    */

  for (storeEntryIdx <- 0 until config.fifoDepth) {
    when(initBits(storeEntryIdx)) {
      storeCompleted(storeEntryIdx) := false.B
    }.elsewhen(io.storeQIdxInValid && io.storeQIdxIn === storeEntryIdx.U) {
      storeCompleted(storeEntryIdx) := true.B
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Communicating with store ports
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** For each store port, set flags denoting its associated store queue entries
    */
  val entriesPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numStorePorts, Vec(config.fifoDepth, Bool()))
  )
  for (storePortId <- 0 until config.numStorePorts) {
    for (storeEntryIdx <- 0 until config.fifoDepth) {
      entriesPorts(storePortId)(storeEntryIdx) := portQ(
        storeEntryIdx
      ) === storePortId.U
    }
  }

  /** Priorities for each store address (or data) port I.e., when an address (or
    * data) port has multiple corresponding stores, the priority is given to the
    * first entry from the head for which the address (or data) is not known
    */
  val inputAddrPriorityPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numStorePorts, Vec(config.fifoDepth, Bool()))
  )
  val inputDataPriorityPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numStorePorts, Vec(config.fifoDepth, Bool()))
  )
  for (storePortId <- 0 until config.numStorePorts) {
    val condVecAddr = VecInit(
      (entriesPorts(storePortId) zip addrKnown) map (x => x._1 && !x._2)
    )
    val condVecData = VecInit(
      (entriesPorts(storePortId) zip dataKnown) map (x => x._1 && !x._2)
    )
    inputAddrPriorityPorts(storePortId) := CyclicPriorityEncoderOH(
      condVecAddr,
      dummyHead
    )
    inputDataPriorityPorts(storePortId) := CyclicPriorityEncoderOH(
      condVecData,
      dummyHead
    )
  }

  /** Update address known and data known flags (at clock) They are cleared when
    * first allocated Then, when the corresponding addresses and data arrive,
    * they are set
    */
  for (storeEntryIdx <- 0 until config.fifoDepth) {
    when(initBits(storeEntryIdx)) {
      addrKnown(storeEntryIdx) := false.B
      dataKnown(storeEntryIdx) := false.B
    }.otherwise {
      val condVecAddr = VecInit(
        Range(0, config.numStorePorts) map (idx =>
          inputAddrPriorityPorts(idx)(storeEntryIdx) && !addrKnown(
            storeEntryIdx
          ) && io.storeAddrEnable(idx)
        )
      )
      when(condVecAddr.exists(x => x)) {
        addrQ(storeEntryIdx) := io.addressFromStorePorts(OHToUInt(condVecAddr))
        addrKnown(storeEntryIdx) := true.B
      }
      val condVecData = VecInit(
        Range(0, config.numStorePorts) map (idx =>
          inputDataPriorityPorts(idx)(storeEntryIdx) && !dataKnown(
            storeEntryIdx
          ) && io.storeDataEnable(idx)
        )
      )
      when(condVecData.exists(x => x)) {
        dataQ(storeEntryIdx) := io.dataFromStorePorts(OHToUInt(condVecData))
        dataKnown(storeEntryIdx) := true.B
      }
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Head and tail update
    * -----------------------------------------------------------------------------------------------------------------
    */

  when(io.storeQIdxOut.ready && storeRequest) {
    dummyHead := addMod(dummyHead, 1.U, config.fifoDepth.U)
  }

  when(
    (io.storeQIdxInValid && io.storeQIdxIn === head) || storeCompleted(head)
  ) {
    head := addMod(head, 1.U, config.fifoDepth.U)
  }

  when(io.bbStart) {
    tail := addMod(tail, io.bbNumStores, config.fifoDepth.U)
  }

  io.storeEmpty := VecInit(
    (storeCompleted zip allocatedEntries) map (x => x._1 || !x._2)
  ).forall(x => x)

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Remaining IO Mapping
    * -----------------------------------------------------------------------------------------------------------------
    */

  io.storeHead := dummyHead
  io.storeTail := tail
  io.storeAddrQueue := addrQ
  io.storeDataQueue := dataQ
  io.storeDataDone := dataKnown
  io.storeAddrDone := addrKnown
  io.storeQIdxOut.valid := storeRequest
  io.storeDataToMem := dataQ(dummyHead)
  io.storeAddrToMem := addrQ(dummyHead)
}
