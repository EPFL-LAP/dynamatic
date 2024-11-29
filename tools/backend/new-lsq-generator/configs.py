import math
import json
import sys

# read and parse the json file
def GetConfigs(path: str):
    with open(path, 'r') as file:
        configsString = file.read()
        # configs_list_raw = json.loads(configsString)["specifications"]
        configs_list = []
        for configs in json.loads(configsString)["specifications"]:
           configs_list.append(Configs(configs))
        return configs_list

class Configs:
  name:          str = 'test'
  dataW:         int = 32
  addrW:         int = 32
  idW:           int = 3
  numLdqEntries: int = 8
  numStqEntries: int = 8
  numLdPorts:    int = 1
  numStPorts:    int = 1
  numGroups:     int = 1
  numLdMem:      int = 1
  numStMem:      int = 1

  stResp:        bool = False
  gaMulti:       bool = False
  
  gaNumLoads:    list = [1]
  gaNumStores:   list = [1]
  gaLdOrder:     list = [[0]]
  gaLdPortIdx:   list = [[0]]
  gaStPortIdx:   list = [[0]]

  ldqAddrW:      int = 3
  stqAddrW:      int = 3
  ldpAddrW:      int = 0
  stpAddrW:      int = 0

  pipe0:        bool = False
  pipe1:        bool = False
  pipeComp:     bool = False
  headLag:      bool = False

  def __init__(self, config: dict) -> None:
    self.name          = config["name"]
    self.dataW         = config["dataW"]
    self.addrW         = config["addrW"]
    self.idW           = config["idW"]
    self.numLdqEntries = config["numLdqEntries"]
    self.numStqEntries = config["numStqEntries"]
    self.numLdPorts    = config["numLdPorts"]
    self.numStPorts    = config["numStPorts"]
    self.numGroups     = config["numGroups"]
    self.numLdMem      = config["numLdMem"]
    self.numStMem      = config["numStMem"]

    self.stResp        = bool(config["stResp"])
    self.gaMulti       = bool(config["gaMulti"])

    self.gaNumLoads    = config["gaNumLoads"]
    self.gaNumStores   = config["gaNumStores"]
    self.gaLdOrder     = config["gaLdOrder"]
    self.gaLdPortIdx   = config["gaLdPortIdx"]
    self.gaStPortIdx   = config["gaStPortIdx"]

    self.ldqAddrW      = math.ceil(math.log2(self.numLdqEntries))
    self.stqAddrW      = math.ceil(math.log2(self.numStqEntries))
    self.emptyLdAddrW  = math.ceil(math.log2(self.numLdqEntries+1))
    self.emptyStAddrW  = math.ceil(math.log2(self.numStqEntries+1))
    self.ldpAddrW      = math.ceil(math.log2(self.numLdPorts))
    self.stpAddrW      = math.ceil(math.log2(self.numStPorts))

    self.pipe0         = bool(config["pipe0"])
    self.pipe1         = bool(config["pipe1"])
    self.pipeComp      = bool(config["pipeComp"])
    self.headLag       = bool(config["headLag"])

    assert(self.idW >= self.ldqAddrW)

    # list size checking
    assert(len(self.gaNumLoads)  == self.numGroups)
    assert(len(self.gaNumStores) == self.numGroups)
    assert(len(self.gaLdOrder)   == self.numGroups)
    assert(len(self.gaLdPortIdx) == self.numGroups)
    assert(len(self.gaStPortIdx) == self.numGroups)

    # for idx in range(0, self.numGroups):
    #     assert(len(self.gaLdOrder[idx])   == self.gaNumLoads[idx])
    #     assert(len(self.gaLdPortIdx[idx]) == self.gaNumLoads[idx])
    #     assert(len(self.gaStPortIdx[idx]) == self.gaNumStores[idx])

if __name__ == '__main__':
    path_configs = './configs/test_new.json'
    configs_list = GetConfigs(path_configs)
    for configs in configs_list:
      print(configs.__dict__)