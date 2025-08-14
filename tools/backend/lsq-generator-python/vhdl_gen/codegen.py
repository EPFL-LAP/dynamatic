from vhdl_gen.context import VHDLContext
import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers
import vhdl_gen.generators.lsq as lsq

import vhdl_gen.generators.lsq_submodule_wrapper as lsq_submodule_wrapper


def codeGen(path_rtl, configs):
    ctx = VHDLContext()

    # Initialize a wrapper object to hold all submodule generator instances.
    lsq_submodules = lsq_submodule_wrapper.LSQ_Submodules()

    name = configs.name + '_core'
    # empty the file
    file = open(f'{path_rtl}/{name}.vhd', 'w').close()

    # Group Allocator
    ga = group_allocator.GroupAllocator(
        name=name, suffix='_ga', configs=configs)
    ga.generate(path_rtl)
    lsq_submodules.group_allocator = ga

    # Load Address Port Dispatcher
    ptq_dispatcher_lda = dispatchers.PortToQueueDispatcher(
        name, '_lda', configs.numLdPorts, configs.numLdqEntries, configs.addrW, configs.ldpAddrW)
    ptq_dispatcher_lda.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_lda = ptq_dispatcher_lda

    # Load Data Port Dispatcher
    qtp_dispatcher_ldd = dispatchers.QueueToPortDispatcher(
        name, '_ldd', configs.numLdPorts, configs.numLdqEntries, configs.dataW, configs.ldpAddrW)
    qtp_dispatcher_ldd.generate(path_rtl)
    lsq_submodules.qtp_dispatcher_ldd = qtp_dispatcher_ldd

    # Store Address Port Dispatcher
    ptq_dispatcher_sta = dispatchers.PortToQueueDispatcher(
        name, '_sta', configs.numStPorts, configs.numStqEntries, configs.addrW, configs.stpAddrW)
    ptq_dispatcher_sta.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_sta = ptq_dispatcher_sta

    # Store Data Port Dispatcher
    ptq_dispatcher_std = dispatchers.PortToQueueDispatcher(
        name, '_std', configs.numStPorts, configs.numStqEntries, configs.dataW, configs.stpAddrW)
    ptq_dispatcher_std.generate(path_rtl)
    lsq_submodules.ptq_dispatcher_std = ptq_dispatcher_std

    # Store Backward Port Dispatcher
    if configs.stResp:
        qtp_dispatcher_stb = dispatchers.QueueToPortDispatcher(
            name, '_stb', configs.numStPorts, configs.numStqEntries, 0, configs.stpAddrW)
        qtp_dispatcher_stb.generate(path_rtl)
        lsq_submodules.qtp_dispatcher_stb = qtp_dispatcher_stb

    # Change the name of the following module to lsq_core
    lsq_core = lsq.LSQ(name, '', configs)
    lsq_core.generate(lsq_submodules, path_rtl)
