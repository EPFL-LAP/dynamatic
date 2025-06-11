from vhdl_gen.context import VHDLContext
import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers
import vhdl_gen.generators.lsq as lsq

from vhdl_gen.generators.registry import register_registry


def codeGen(path_rtl, configs):
    ctx = VHDLContext()

    name = configs.name + '_core'
    # empty the file
    file = open(f'{path_rtl}/{name}.vhd', 'w').close()

    # Group Allocator
    ga = group_allocator.GroupAllocator(ctx, path_rtl, name, '_ga', configs)
    ga.generate()
    register_registry(ga)

    # Load Address Port Dispatcher
    ptq_dispatcher_lda = dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_lda', configs.numLdPorts, configs.numLdqEntries, configs.addrW, configs.ldpAddrW)
    ptq_dispatcher_lda.generate()
    register_registry(ptq_dispatcher_lda)

    # Load Data Port Dispatcher
    qtp_dispatcher_ldd = dispatchers.QueueToPortDispatcher(ctx, path_rtl, name, '_ldd', configs.numLdPorts, configs.numLdqEntries, configs.dataW, configs.ldpAddrW)
    qtp_dispatcher_ldd.generate()
    register_registry(qtp_dispatcher_ldd)

    # Store Address Port Dispatcher
    ptq_dispatcher_sta = dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_sta', configs.numStPorts, configs.numStqEntries, configs.addrW, configs.stpAddrW)
    ptq_dispatcher_sta.generate()
    register_registry(ptq_dispatcher_sta)

    # Store Data Port Dispatcher
    ptq_dispatcher_std = dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_std', configs.numStPorts, configs.numStqEntries, configs.dataW, configs.stpAddrW)
    ptq_dispatcher_std.generate()
    register_registry(ptq_dispatcher_std)

    # Store Backward Port Dispatcher
    if configs.stResp:
        qtp_dispatcher_stb = dispatchers.QueueToPortDispatcher(ctx, path_rtl, name, '_stb', configs.numStPorts, configs.numStqEntries, 0, configs.stpAddrW)
        qtp_dispatcher_stb.generate()
        register_registry(qtp_dispatcher_stb)

    # Change the name of the following module to lsq_core
    lsq_core = lsq.LSQ(ctx, path_rtl, name, '', configs)
    lsq_core.generate()


