from generators.support.utils import *


def generate_store(name, params):
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    addr_type = SmvScalarType(params[ATTR_ADDR_BITWIDTH])

    return _generate_store(name, data_type, addr_type)


def _generate_store(name, data_type, addr_type):
    return f"""
MODULE {name}(addrIn, addrIn_valid, dataIn, dataIn_valid, addrOut_ready, dataToMem_ready)

  //output
  DEFINE
  dataIn_ready := dataToMem_ready;
  addrIn_ready := addrOut_ready;
  dataToMem := dataIn;
  dataToMem_valid := dataIn_valid;
  addrOut := addrIn;
  addrOut_valid := addrIn_valid;
"""
