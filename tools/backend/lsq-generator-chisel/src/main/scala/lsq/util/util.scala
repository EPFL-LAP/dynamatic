//===- util.scala ---------------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq

import chisel3._
import chisel3.util.{Mux1H, OHToUInt, PriorityEncoderOH, UIntToOH}
import lsq.config.LsqConfigs

package object util {

  def addMod(a: UInt, b: UInt, modVal: UInt): UInt = {
    (a +& b) % modVal
  }

  def subMod(a: UInt, b: UInt, modVal: UInt): UInt = {
    ((modVal +& a) - b) % modVal
  }

  class DataAddressPair(config: LsqConfigs) extends Bundle {
    val data = UInt(config.dataWidth.W)
    val address = UInt(config.addrWidth.W)
  }

  /** Given a Bool vector and an offset, computes the OH encoding of the first
    * True value starting from offset
    */
  object CyclicPriorityEncoderOH {

    def apply(signals: Vec[Bool], offset: Int): Vec[Bool] = {
      val temp = PriorityEncoderOH(signals.drop(offset) ++ signals.take(offset))
      VecInit(temp.takeRight(offset) ++ temp.dropRight(offset))
    }

    def apply(signals: Vec[Bool], offset: UInt): Vec[Bool] = {
      Mux1H(
        VecInit[Bool](UIntToOH(offset, signals.length).asBools).zipWithIndex map
          (x => (x._1, CyclicPriorityEncoderOH(signals, x._2)))
      )
    }

  }

  /** Given a Bool vector and an offset, computes the index of the first True
    * value starting from offset
    */
  object CyclicPriorityEncoder {

    def apply(signals: Vec[Bool], offset: UInt): UInt = {
      OHToUInt(CyclicPriorityEncoderOH(signals, offset))
    }

  }

  /** Cyclically right shift a given vector by a specified amount
    */
  object CyclicLeftShift {

    def apply[T <: Data](signals: Vec[T], offset: Int): Vec[T] = {
      VecInit(signals.drop(offset) ++ signals.take(offset))
    }

    def apply[T <: Data](signals: Vec[T], offset: UInt): Vec[T] = {
      Mux1H(
        VecInit[Bool](UIntToOH(offset, signals.length).asBools).zipWithIndex map
          (x => (x._1, CyclicLeftShift(signals, x._2)))
      )
    }

  }

  /** Cyclically right shift a given vector by a specified amount
    */
  object CyclicRightShift {

    def apply[T <: Data](signals: Vec[T], offset: Int): Vec[T] = {
      VecInit(signals.takeRight(offset) ++ signals.dropRight(offset))
    }

    def apply[T <: Data](signals: Vec[T], offset: UInt): Vec[T] = {
      Mux1H(
        VecInit[Bool](UIntToOH(offset, signals.length).asBools).zipWithIndex map
          (x => (x._1, CyclicRightShift(signals, x._2)))
      )
    }

  }

}
