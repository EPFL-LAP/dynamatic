from vhdl_gen.context import VHDLContext
import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers
import vhdl_gen.generators.lsq as lsq


def codeGen(path_rtl, configs):
    ctx = VHDLContext()

    name = configs.name
    # empty the file
    file = open(f'{path_rtl}/{name}_core.vhd', 'w').close()
    # Group Allocator
    group_allocator.GroupAllocator(ctx, path_rtl, name, '_core_ga', configs)
    # Load Address Port Dispatcher
    dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_core_lda',
                                      configs.numLdPorts, configs.numLdqEntries, configs.addrW, configs.ldpAddrW
                                      )
    # Load Data Port Dispatcher
    dispatchers.QueueToPortDispatcher(ctx, path_rtl, name, '_core_ldd',
                                      configs.numLdPorts, configs.numLdqEntries, configs.dataW, configs.ldpAddrW
                                      )
    # Store Address Port Dispatcher
    dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_core_sta',
                                      configs.numStPorts, configs.numStqEntries, configs.addrW, configs.stpAddrW
                                      )
    # Store Data Port Dispatcher
    dispatchers.PortToQueueDispatcher(ctx, path_rtl, name, '_core_std',
                                      configs.numStPorts, configs.numStqEntries, configs.dataW, configs.stpAddrW
                                      )
    # Store Backward Port Dispatcher
    if configs.stResp:
        dispatchers.QueueToPortDispatcher(ctx, path_rtl, name, '_core_stb',
                                          configs.numStPorts, configs.numStqEntries, 0, configs.stpAddrW
                                          )

    # Change the name of the following module to lsq_core
    lsq.LSQ(ctx, path_rtl, name + '_core', configs)
