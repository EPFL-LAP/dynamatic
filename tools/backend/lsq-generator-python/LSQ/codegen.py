from LSQ.context import VHDLContext
from LSQ.generators.group_allocator.group_allocator import get_group_allocator, GroupAllocator
import LSQ.generators.dispatchers as dispatchers
import LSQ.generators.core as core

import LSQ.generators.lsq_submodule_wrapper as lsq_submodule_wrapper

from LSQ.config import Config

def codeGen(path_rtl, config : Config):
    # Initialize a wrapper object to hold all submodule generator instances.
    lsq_submodules = lsq_submodule_wrapper.LSQ_Submodules()

    name = config.name + '_core'

    # with open(f'{path_rtl}/{name}.vhd', 'a') as file:
    #     file.write(get_group_allocator(config, name))

    ga = GroupAllocator(name, "_ga", config)
    ga.generate(path_rtl)

    # Load Address Port Dispatcher
    ptq_dispatcher_lda = dispatchers.PortToQueueDispatcher(
        name, '_lda', config.numLdPorts, config.numLdqEntries, config.addrW, config.ldpAddrW)
    ptq_dispatcher_lda.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_lda = ptq_dispatcher_lda

    # Load Data Port Dispatcher
    qtp_dispatcher_ldd = dispatchers.QueueToPortDispatcher(
        name, '_ldd', config.numLdPorts, config.numLdqEntries, config.payload_bitwidth, config.ldpAddrW)
    qtp_dispatcher_ldd.generate(path_rtl)
    lsq_submodules.qtp_dispatcher_ldd = qtp_dispatcher_ldd

    # Store Address Port Dispatcher
    ptq_dispatcher_sta = dispatchers.PortToQueueDispatcher(
        name, '_sta', config.numStPorts, config.numStqEntries, config.addrW, config.stpAddrW)
    ptq_dispatcher_sta.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_sta = ptq_dispatcher_sta

    # Store Data Port Dispatcher
    ptq_dispatcher_std = dispatchers.PortToQueueDispatcher(
        name, '_std', config.numStPorts, config.numStqEntries, config.payload_bitwidth, config.stpAddrW)
    ptq_dispatcher_std.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_std = ptq_dispatcher_std

    # Store Backward Port Dispatcher
    if config.stResp:
        qtp_dispatcher_stb = dispatchers.QueueToPortDispatcher(
            name, '_stb', config.numStPorts, config.numStqEntries, 0, config.stpAddrW)
        qtp_dispatcher_stb.generate(path_rtl)
        lsq_submodules.qtp_dispatcher_stb = qtp_dispatcher_stb

    # Change the name of the following module to lsq_core
    lsq_core = core.LSQ(name, '', config)
    lsq_core.generate(lsq_submodules, path_rtl)
