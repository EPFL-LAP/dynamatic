from generators.support.utils import *


def generate_store(name, params):
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["dataIn"])
  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["addrIn"])

  return _generate_store(name, data_type, addr_type)


def _generate_store(name, data_type, addr_type):
  return f"""
MODULE {name}(dataIn, dataIn_valid, addrIn, addrIn_valid, dataToMem_ready, addrOut_ready)

  //output
  DEFINE
  dataIn_ready := dataToMem_ready;
  addrIn_ready := addrOut_ready;
  dataToMem := dataIn;
  dataToMem_valid := dataIn_valid;
  addrOut := addrIn;
  addrOut_valid := addrIn_valid;
"""
