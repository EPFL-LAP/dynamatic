//===- LoadQueue.scala ----------------------------------------*- Scala -*-===//
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

class LoadQueue(config: LsqConfigs) extends Module {
  override def desiredName: String = "LOAD_QUEUE_" + config.name

  val io = IO(new Bundle {
    // From group allocator
    val bbStart = Input(Bool())
    val bbLoadOffsets =
      Input(Vec(config.fifoDepth_L, UInt(log2Ceil(config.fifoDepth_S).W)))
    val bbLoadPorts =
      Input(Vec(config.fifoDepth_L, UInt(config.loadPortIdWidth)))
    val bbNumLoads = Input(
      UInt(max(1, log2Ceil(min(config.numLoadPorts, config.fifoDepth_L) + 1)).W)
    )
    // To group allocator
    val loadTail = Output(UInt(log2Ceil(config.fifoDepth_L).W))
    val loadHead = Output(UInt(log2Ceil(config.fifoDepth_L).W))
    val loadEmpty = Output(Bool())
    // From store queue
    val storeTail = Input(UInt(log2Ceil(config.fifoDepth_S).W))
    val storeHead = Input(UInt(log2Ceil(config.fifoDepth_S).W))
    val storeEmpty = Input(Bool())
    val storeAddrDone = Input(Vec(config.fifoDepth_S, Bool()))
    val storeDataDone = Input(Vec(config.fifoDepth_S, Bool()))
    val storeAddrQueue =
      Input(Vec(config.fifoDepth_S, UInt(config.addrWidth.W)))
    val storeDataQueue =
      Input(Vec(config.fifoDepth_S, UInt(config.dataWidth.W)))
    // To store queue
    val loadAddrDone = Output(Vec(config.fifoDepth_L, Bool()))
    val loadDataDone = Output(Vec(config.fifoDepth_L, Bool()))
    val loadAddrQueue =
      Output(Vec(config.fifoDepth_L, UInt(config.addrWidth.W)))
    // Interface to load ports
    val loadAddrEnable = Input(Vec(config.numLoadPorts, Bool()))
    val addrFromLoadPorts =
      Input(Vec(config.numLoadPorts, UInt(config.addrWidth.W)))
    val loadPorts: Vec[DecoupledIO[UInt]] =
      Vec(config.numLoadPorts, Decoupled(UInt(config.dataWidth.W)))
    // Interface to memory
    val loadDataFromMem = Input(UInt(config.dataWidth.W))
    val loadAddrToMem = Output(UInt(config.addrWidth.W))
    val loadEnableToMem = Output(Bool())
    val memIsReadyForLoads = Input(Bool())

  })

  require(config.fifoDepth_L > 1)

  val head = RegInit(0.U(log2Ceil(config.fifoDepth_L).W))
  val tail = RegInit(0.U(log2Ceil(config.fifoDepth_L).W))

  val offsetQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth_L)(0.U(config.fifoIdxWidth_S)))
  )
  val portQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth_L)(0.U(config.loadPortIdWidth)))
  )
  val addrQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth_L)(0.U(config.addrWidth.W)))
  )
  val dataQ = RegInit(
    VecInit(Seq.fill(config.fifoDepth_L)(0.U(config.dataWidth.W)))
  )
  val addrKnown = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))
  val dataKnown = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))
  val loadCompleted = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))
  val allocatedEntries = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))
  val bypassInitiated = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))
  val checkBits = RegInit(VecInit(Seq.fill(config.fifoDepth_L)(false.B)))

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Allocating a new BB
    * -----------------------------------------------------------------------------------------------------------------
    */

  val initBits = VecInit(
    Range(0, config.fifoDepth_L) map (idx =>
      (subMod(idx.U, tail, config.fifoDepth_L.U) < io.bbNumLoads) && io.bbStart
    )
  )

  allocatedEntries := VecInit(
    (allocatedEntries zip initBits) map (x => x._1 || x._2)
  )

  for (idx <- 0 until config.fifoDepth_L) {
    when(initBits(idx)) {
      offsetQ(idx) := io.bbLoadOffsets(
        subMod(idx.U, tail, config.fifoDepth_L.U)
      )
      portQ(idx) := io.bbLoadPorts(subMod(idx.U, tail, config.fifoDepth_L.U))
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Find Conflicts
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** Computes whether any load should be checked for conflicts (at clock) No
    * need to check an entry if all stores up to the last preceding store are
    * already executed I.e., store head has gone past the last preceding store
    */
  val previousStoreHead = RegNext(io.storeHead)
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    when(initBits(loadEntryIdx)) {
      checkBits(loadEntryIdx) := !(io.storeEmpty && addMod(
        io.bbLoadOffsets(subMod(loadEntryIdx.U, tail, config.fifoDepth_L.U)),
        1.U,
        config.fifoDepth_S.U
      ) === io.storeTail)
    }.otherwise {
      when(io.storeEmpty) {
        checkBits(loadEntryIdx) := false.B
      }.elsewhen(
        previousStoreHead <= offsetQ(loadEntryIdx) && offsetQ(
          loadEntryIdx
        ) < io.storeHead
      ) {
        checkBits(loadEntryIdx) := false.B
      }.elsewhen(
        previousStoreHead > io.storeHead && !(io.storeHead <= offsetQ(
          loadEntryIdx
        ) &&
          offsetQ(loadEntryIdx) < previousStoreHead)
      ) {
        checkBits(loadEntryIdx) := false.B
      }
    }
  }

  /** Shifted store queue entries and corresponding flags
    */
  val shiftedStoreAddrQ = CyclicLeftShift(io.storeAddrQueue, io.storeHead)
  val shiftedStoreAddrKnown = CyclicLeftShift(io.storeAddrDone, io.storeHead)
  val shiftedStoreDataQ = CyclicLeftShift(io.storeDataQueue, io.storeHead)
  val shiftedStoreDataKnown = CyclicLeftShift(io.storeDataDone, io.storeHead)

  /** Entries in the store queue are valid if they are between the store head
    * and the store tail
    */
  val validEntriesInStoreQ: Vec[Bool] = VecInit(
    Range(0, config.fifoDepth_S) map (idx =>
      Mux(
        io.storeHead < io.storeTail,
        io.storeHead <= idx.U && idx.U < io.storeTail,
        !io.storeEmpty && !(io.storeTail <= idx.U && idx.U < io.storeHead)
      )
    )
  )

  /** Store entries should be checked only if they are between the store head
    * and the last preceding store (i.e., offsetQ(head))
    */
  val storesToCheck: Vec[Vec[Bool]] = Wire(
    Vec(config.fifoDepth_L, Vec(config.fifoDepth_S, Bool()))
  )
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    storesToCheck(loadEntryIdx) := VecInit(
      Range(0, config.fifoDepth_S) map (idx =>
        Mux(
          io.storeHead <= offsetQ(loadEntryIdx),
          io.storeHead <= idx.U && idx.U <= offsetQ(loadEntryIdx),
          !(offsetQ(loadEntryIdx) < idx.U && idx.U < io.storeHead)
        )
      )
    )
  }

  /** For each load, only the following entries in the storeQ should be checked
    * for conflicts
    */
  val entriesToCheck: Vec[Vec[Bool]] = Wire(
    Vec(config.fifoDepth_L, Vec(config.fifoDepth_S, Bool()))
  )
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    entriesToCheck(loadEntryIdx) := (storesToCheck(
      loadEntryIdx
    ) zip validEntriesInStoreQ) map
      (x => x._1 && x._2 && checkBits(loadEntryIdx))
  }

  /** Checking conflicts for each load Conflict occurs when store address of a
    * store entry is known and is equal to the load address of a load entry
    */
  val conflict: Vec[Vec[Bool]] = Wire(
    Vec(config.fifoDepth_L, Vec(config.fifoDepth_S, Bool()))
  )
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    for (storeEntryIdx <- 0 until config.fifoDepth_S) {
      conflict(loadEntryIdx)(storeEntryIdx) := entriesToCheck(loadEntryIdx)(
        storeEntryIdx
      ) &&
        io.storeAddrDone(storeEntryIdx) && addrKnown(loadEntryIdx) &&
        (addrQ(loadEntryIdx) === io.storeAddrQueue(storeEntryIdx))
    }
  }

  /** Keep track of the store entries where the store address is not known
    */
  val storeAddrNotKnownFlags: Vec[Vec[Bool]] = Wire(
    Vec(config.fifoDepth_L, Vec(config.fifoDepth_S, Bool()))
  )
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    storeAddrNotKnownFlags(
      loadEntryIdx
    ) := (io.storeAddrDone zip entriesToCheck(loadEntryIdx) map
      (x => !x._1 && x._2))
  }

  val conflictPReg = RegNext(
    VecInit(conflict map (x => CyclicLeftShift(x, io.storeHead)))
  )
  val storeAddrNotKnownFlagsPReg = RegNext(
    VecInit(storeAddrNotKnownFlags map (x => CyclicLeftShift(x, io.storeHead)))
  )
  val shiftedStoreDataKnownPReg = RegNext(shiftedStoreDataKnown)
  val shiftedStoreDataQPreg = RegNext(shiftedStoreDataQ)
  val addrKnownPReg = RegNext(addrKnown)
  val dataKnownPReg = RegNext(dataKnown)

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Load / Bypass Data
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** For each load, find the last conflicting store entry preceding that Also
    * compute whether it may be bypassed (i.e., its new data value is known)
    */
  val lastConflict: Vec[Vec[Bool]] = Wire(
    Vec(config.fifoDepth_L, Vec(config.fifoDepth_S, Bool()))
  )
  val canBypass: Vec[Bool] = Wire(Vec(config.fifoDepth_L, Bool()))
  val bypassVal: Vec[UInt] = Wire(Vec(config.fifoDepth_L, UInt()))
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    lastConflict(loadEntryIdx) := VecInit(Seq.fill(config.fifoDepth_S)(false.B))
    canBypass(loadEntryIdx) := false.B
    bypassVal(loadEntryIdx) := 0.U(config.dataWidth.W)
    val lastIdx =
      conflictPReg(loadEntryIdx).lastIndexWhere((x: Bool) => x === true.B)
    when(conflictPReg(loadEntryIdx).exists(x => x)) {
      lastConflict(loadEntryIdx)(lastIdx) := true.B
      canBypass(loadEntryIdx) := shiftedStoreDataKnownPReg(lastIdx)
      bypassVal(loadEntryIdx) := shiftedStoreDataQPreg(lastIdx)
    }
  }

  val loadRequest: Vec[Bool] = Wire(Vec(config.fifoDepth_L, Bool()))
  val bypassRequest: Vec[Bool] = Wire(Vec(config.fifoDepth_L, Bool()))

  /** When multiple load requests, the first request starting from the load head
    * gets the priority
    */
  val priorityLoadRequest = CyclicPriorityEncoderOH(loadRequest, head)

  val prevPriorityRequest = RegInit(
    VecInit(Seq.fill(config.fifoDepth_L)(false.B))
  )
  when(io.memIsReadyForLoads) {
    prevPriorityRequest := priorityLoadRequest
  }.otherwise {
    prevPriorityRequest := VecInit(Seq.fill(config.fifoDepth_L)(false.B))
  }

  for (i <- 0 until config.fifoDepth_L) {
    when(initBits(i)) {
      bypassInitiated(i) := false.B
    }.elsewhen(bypassRequest(i)) {
      bypassInitiated(i) := true.B
    }
  }

  /** Decide which loads can progress (i.e., no possible conflicts) and which
    * loads can be bypassed
    */
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    loadRequest(loadEntryIdx) := false.B
    bypassRequest(loadEntryIdx) := false.B
    // consider only the loads for which data is not known but the address is known
    when(addrKnownPReg(loadEntryIdx) && !dataKnownPReg(loadEntryIdx)) {
      when(
        !bypassInitiated(loadEntryIdx) && !prevPriorityRequest(
          loadEntryIdx
        ) && !dataKnown(loadEntryIdx)
      ) {
        // load can happen if we are sure that there are no conflicts
        // i.e., addresses of all necessary entries are known and they do not conflict with the load address
        loadRequest(loadEntryIdx) := !storeAddrNotKnownFlagsPReg(loadEntryIdx)
          .exists((x: Bool) => x) &&
          !conflictPReg(loadEntryIdx).exists((x: Bool) => x)
        // load can be bypassed if canBypass is true for the last conflicting store and all addresses for stores up to
        // the last preceding store are known
        bypassRequest(loadEntryIdx) := canBypass(loadEntryIdx) &&
          storeAddrNotKnownFlagsPReg(loadEntryIdx).asUInt < lastConflict(
            loadEntryIdx
          ).asUInt
      }
    }
  }

  /** Sending the priority load request to memory
    */
  io.loadEnableToMem := false.B
  when(loadRequest.exists((x: Bool) => x)) {
    io.loadAddrToMem := addrQ(PriorityEncoder(priorityLoadRequest))
    io.loadEnableToMem := true.B
  }.otherwise {
    io.loadAddrToMem := 0.U
    io.loadEnableToMem := false.B
  }

  /** Data is known if either memory returns it (after 1-cycle delay in case of
    * BRAM) or it is bypassed from store queue
    */
  for (i <- 0 until config.fifoDepth_L) {
    when(initBits(i)) {
      dataKnown(i) := false.B
    }.elsewhen(prevPriorityRequest(i) || bypassRequest(i)) {
      dataKnown(i) := true.B
    }
  }

  for (idx <- 0 until config.fifoDepth_L) {
    when(bypassRequest(idx)) {
      dataQ(idx) := bypassVal(idx)
    }.elsewhen(prevPriorityRequest(idx)) {
      dataQ(idx) := io.loadDataFromMem
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Communicating with load ports
    * -----------------------------------------------------------------------------------------------------------------
    */

  /** For each load port, set flags denoting its associated load queue entries
    */
  val entriesPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numLoadPorts, Vec(config.fifoDepth_L, Bool()))
  )
  for (loadPortId <- 0 until config.numLoadPorts) {
    for (loadEntryIdx <- 0 until config.fifoDepth_L) {
      entriesPorts(loadPortId)(loadEntryIdx) := portQ(
        loadEntryIdx
      ) === loadPortId.U
    }
  }

  /** Priorities for each load port When a port has multiple corresponding
    * loads, the input priority is given to the first entry from the head for
    * which the address is not known. The output priority is given to the first
    * entry from the head
    */
  val inputPriorityPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numLoadPorts, Vec(config.fifoDepth_L, Bool()))
  )
  val outputPriorityPorts: Vec[Vec[Bool]] = Wire(
    Vec(config.numLoadPorts, Vec(config.fifoDepth_L, Bool()))
  )
  for (loadPortId <- 0 until config.numLoadPorts) {
    val condVec = VecInit(
      (entriesPorts(loadPortId) zip addrKnown) map (x => x._1 && !x._2)
    )
    inputPriorityPorts(loadPortId) := CyclicPriorityEncoderOH(condVec, head)
    outputPriorityPorts(loadPortId) := CyclicPriorityEncoderOH(
      entriesPorts(loadPortId),
      head
    )
  }

  /** Update address known flags (at clock) They are cleared when first
    * allocated and set when the corresponding addresses arrive
    */
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    when(initBits(loadEntryIdx)) {
      addrKnown(loadEntryIdx) := false.B
    }.otherwise {
      // At most one bit of condVec is high
      val condVec = VecInit(
        Range(0, config.numLoadPorts) map (idx =>
          inputPriorityPorts(idx)(loadEntryIdx) && io.loadAddrEnable(idx)
        )
      )
      when(condVec.exists(x => x)) {
        addrQ(loadEntryIdx) := io.addrFromLoadPorts(OHToUInt(condVec))
        addrKnown(loadEntryIdx) := true.B
      }
    }
  }

  /** Computes whether a load is completed at the next positive clock edge
    */
  val loadCompleting: Vec[Bool] = Wire(Vec(config.fifoDepth_L, Bool()))
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    val condVec = VecInit(
      Range(0, config.numLoadPorts) map (idx =>
        outputPriorityPorts(idx)(loadEntryIdx) &&
          dataKnown(loadEntryIdx) && !loadCompleted(loadEntryIdx) && io
            .loadPorts(idx)
            .ready
      )
    )
    loadCompleting(loadEntryIdx) := condVec.exists(x => x)
  }

  /** Updating completed flags for loads (at clock)
    */
  for (loadEntryIdx <- 0 until config.fifoDepth_L) {
    when(initBits(loadEntryIdx)) {
      loadCompleted(loadEntryIdx) := false.B
    }.elsewhen(loadCompleting(loadEntryIdx)) {
      loadCompleted(loadEntryIdx) := true.B
    }
  }

  /** Sending data to load ports
    */
  for (loadPortId <- 0 until config.numLoadPorts) {
    val condVec = VecInit(
      Range(0, config.fifoDepth_L) map
        (idx =>
          outputPriorityPorts(loadPortId)(idx) && dataKnown(
            idx
          ) && !loadCompleted(idx)
        )
    )
    when(condVec.exists(x => x)) {
      io.loadPorts(loadPortId).bits := dataQ(PriorityEncoder(condVec))
      io.loadPorts(loadPortId).valid := true.B
    }.otherwise {
      io.loadPorts(loadPortId).bits := 0.U
      io.loadPorts(loadPortId).valid := false.B
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Head and tail update
    * -----------------------------------------------------------------------------------------------------------------
    */

  when(
    (loadCompleted(head) || loadCompleting(
      head
    )) && (head =/= tail || !io.loadEmpty)
  ) {
    head := addMod(head, 1.U, config.fifoDepth_L.U)
  }

  when(io.bbStart) {
    tail := addMod(tail, io.bbNumLoads, config.fifoDepth_L.U)
  }

  io.loadEmpty := VecInit(
    (loadCompleted zip allocatedEntries) map (x => x._1 || !x._2)
  ).forall(x => x)

  /** -----------------------------------------------------------------------------------------------------------------
    * Section: Remaining IO Mapping
    * -----------------------------------------------------------------------------------------------------------------
    */

  io.loadHead := head
  io.loadTail := tail
  io.loadAddrQueue := addrQ
  io.loadAddrDone := addrKnown
  io.loadDataDone := dataKnown
}
