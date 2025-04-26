#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
import json
import sys

# read and parse the json file


def GetConfigs(path: str):
  with open(path, "r") as file:
    configString = file.read()
    configs = json.loads(configString)
    return Configs(configs)


class Configs:
  name: str = "test"
  dataW: int = 32
  addrW: int = 32
  idW: int = 3
  numLdqEntries: int = 8
  numStqEntries: int = 8
  numLdPorts: int = 1
  numStPorts: int = 1
  numGroups: int = 1
  numLdMem: int = 1
  numStMem: int = 1

  stResp: bool = False
  gaMulti: bool = False

  gaNumLoads: list = [1]
  gaNumStores: list = [1]
  gaLdOrder: list = [[0]]
  gaLdPortIdx: list = [[0]]
  gaStPortIdx: list = [[0]]

  ldqAddrW: int = 3
  stqAddrW: int = 3
  ldpAddrW: int = 0
  stpAddrW: int = 0

  pipe0: bool = False
  pipe1: bool = False
  pipeComp: bool = False
  headLag: bool = False

  def __init__(self, config: dict) -> None:
    self.name = config["name"]
    self.dataW = config["dataWidth"]
    self.addrW = config["addrWidth"]
    self.idW = config["indexWidth"]
    self.numLdqEntries = config["fifoDepth_L"]
    self.numStqEntries = config["fifoDepth_S"]
    self.numLdPorts = config["numLoadPorts"]
    self.numStPorts = config["numStorePorts"]
    self.numGroups = config["numBBs"]
    self.numLdMem = config["numLdChannels"]
    self.numStMem = config["numStChannels"]
    self.master = bool(config["master"])

    self.stResp = bool(config["stResp"])
    self.gaMulti = bool(config["groupMulti"])

    self.gaNumLoads = config["numLoads"]
    self.gaNumStores = config["numStores"]
    self.gaLdOrder = config["ldOrder"]
    self.gaLdPortIdx = config["ldPortIdx"]
    self.gaStPortIdx = config["stPortIdx"]

    self.ldqAddrW = math.ceil(math.log2(self.numLdqEntries))
    self.stqAddrW = math.ceil(math.log2(self.numStqEntries))
    self.emptyLdAddrW = math.ceil(math.log2(self.numLdqEntries + 1))
    self.emptyStAddrW = math.ceil(math.log2(self.numStqEntries + 1))
    # Check the number of ports, if num*Ports == 0, set it to 1
    self.ldpAddrW = math.ceil(
        math.log2(self.numLdPorts if self.numLdPorts > 0 else 1)
    )
    self.stpAddrW = math.ceil(
        math.log2(self.numStPorts if self.numStPorts > 0 else 1)
    )

    self.pipe0 = bool(config["pipe0En"])
    self.pipe1 = bool(config["pipe1En"])
    self.pipeComp = bool(config["pipeCompEn"])
    self.headLag = bool(config["headLagEn"])

    assert self.idW >= self.ldqAddrW

    # list size checking
    assert len(self.gaNumLoads) == self.numGroups
    assert len(self.gaNumStores) == self.numGroups
    assert len(self.gaLdOrder) == self.numGroups
    assert len(self.gaLdPortIdx) == self.numGroups
    assert len(self.gaStPortIdx) == self.numGroups
