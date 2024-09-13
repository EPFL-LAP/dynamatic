//===- Main.scala ---------------------------------------------*- Scala -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

package lsq

import chisel3._
import chisel3.util._
import chisel3.stage.ChiselGeneratorAnnotation
import _root_.circt.stage.{ChiselStage, FirtoolOption}
import lsq.config.LsqConfigs
import upickle.default._

object Main extends App {
  // parse command line arguments
  val usage =
    """
    Usage: java -jar [-Xmx7G]  lsq.jar [--target-dir  target_dir]  --spec-file  spec_file.json
    """
  if (args.length == 0) println(usage)
  val arglist = args.toList
  type OptionMap = Map[Symbol, String]

  def nextOption(map: OptionMap, list: List[String]): OptionMap = {
    list match {
      case Nil => map
      case "--spec-file" :: value :: tail =>
        nextOption(map ++ Map(Symbol("specFile") -> value), tail)
      case "--target-dir" :: value :: tail =>
        nextOption(map ++ Map(Symbol("targetFolder") -> value), tail)
      case string :: Nil =>
        nextOption(map ++ Map(Symbol("infile") -> string), list.tail)
      case option :: _ =>
        println("Unknown option " + option)
        map
    }
  }

  val options = nextOption(Map(), arglist)

  if (!options.contains(Symbol("specFile"))) {
    println("--spec-file argument is mandatory!")
    println(usage)
    System.exit(1)
  }

  // read the spec file
  val source = scala.io.Source.fromFile(options(Symbol("specFile")))
  val jsonString =
    try source.mkString
    finally source.close()
  implicit val lsqConfigRW: ReadWriter[LsqConfigs] = macroRW[LsqConfigs]
  val config: LsqConfigs = read[LsqConfigs](jsonString)

  val chiselArgs: Array[String] = Array(
    "--target",
    "verilog",
    "--target-dir",
    options.getOrElse(
      Symbol("targetFolder"),
      config.name + "_" + config.fifoDepth
    )
  )
  val firtoolArgs: Array[String] = Array(
    "--lowering-options=noAlwaysComb,disallowPackedArrays,disallowLocalVariables",
    "--disable-all-randomization"
  )

  if (config.master){
    (new ChiselStage).execute(
      chiselArgs,
      Seq(ChiselGeneratorAnnotation(() => new LSQMaster(config))) ++ firtoolArgs
        .map(
          FirtoolOption(_)
        )
    )
  } else {
    (new ChiselStage).execute(
      chiselArgs,
      Seq(ChiselGeneratorAnnotation(() => new LSQSlave(config))) ++ firtoolArgs
        .map(
          FirtoolOption(_)
        )
    )
  }
}
