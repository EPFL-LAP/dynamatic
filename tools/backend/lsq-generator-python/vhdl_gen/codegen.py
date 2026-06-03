from vhdl_gen.configs import Configs
from vhdl_gen.context import VHDLContext
import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers
import vhdl_gen.generators.bloom_filter as bloom_filter
import vhdl_gen.generators.lsq as lsq

import vhdl_gen.generators.lsq_submodule_wrapper as lsq_submodule_wrapper


def codeGen(path_rtl, configs: Configs):
    ctx = VHDLContext()

    # Initialize a wrapper object to hold all submodule generator instances.
    lsq_submodules = lsq_submodule_wrapper.LSQ_Submodules()

    name = configs.name + '_core'
    # empty the file
    file = open(f'{path_rtl}/{name}.vhd', 'w').close()

    # Group Allocator
    ga = group_allocator.GroupAllocator(
        name=name, suffix='_ga', configs=configs)
    ga.generate(lsq_submodules, path_rtl)
    lsq_submodules.group_allocator = ga

    # Bloom Filter Hash
    if configs.bloomFilterLoad or configs.bloomFilterStore:
        bf_hash = bloom_filter.BloomFilterHash(name, '_bfh', configs)
        bf_hash.generate(lsq_submodules, path_rtl)
        lsq_submodules.bf_hash = bf_hash

    # When the condition "if configs.numLdPorts > 0:" is not true:
    # Do not generating dispatching modules when there are zero load ports.
    #
    # - WARNING: This logic needs more testing
    # - TODO: Also remove the load queue when there are zero load ports.
    if configs.numLdPorts > 0:
        # Load Address Port Dispatcher
        # NOTE: We need to generate all the Bloom filters for the load entries when they
        #       are needed for store issue, hence the use of bloomFilter*Store* here.
        ptq_dispatcher_lda = dispatchers.PortToQueueDispatcher(
            name, '_lda', configs.numLdPorts, configs.numLdqEntries, configs.addrW, configs.ldpAddrW,
            bloomFilter=configs.bloomFilterStore and configs.bloomFilterSequential,
            bloomFilterW=configs.bloomFilterW)
        ptq_dispatcher_lda.generate(lsq_submodules, path_rtl)
        lsq_submodules.ptq_dispatcher_lda = ptq_dispatcher_lda

        # Load Data Port Dispatcher
        qtp_dispatcher_ldd = dispatchers.QueueToPortDispatcher(
            name, '_ldd', configs.numLdPorts, configs.numLdqEntries, configs.dataW, configs.ldpAddrW)
        qtp_dispatcher_ldd.generate(lsq_submodules, path_rtl)
        lsq_submodules.qtp_dispatcher_ldd = qtp_dispatcher_ldd

    # Store Address Port Dispatcher
    # NOTE: We need to generate all the Bloom filters for the store entries when they
    #       are needed for load issue, hence the use of bloomFilter*Load* here.
    ptq_dispatcher_sta = dispatchers.PortToQueueDispatcher(
        name, '_sta', configs.numStPorts, configs.numStqEntries, configs.addrW, configs.stpAddrW,
        bloomFilter=configs.bloomFilterLoad and configs.bloomFilterSequential,
        bloomFilterW=configs.bloomFilterW)
    ptq_dispatcher_sta.generate(lsq_submodules, path_rtl)
    lsq_submodules.ptq_dispatcher_sta = ptq_dispatcher_sta

    # Store Data Port Dispatcher
    ptq_dispatcher_std = dispatchers.PortToQueueDispatcher(
        name, '_std', configs.numStPorts, configs.numStqEntries, configs.dataW, configs.stpAddrW)
    ptq_dispatcher_std.generate(lsq_submodules, path_rtl)
    lsq_submodules.ptq_dispatcher_std = ptq_dispatcher_std

    # Store Backward Port Dispatcher
    if configs.stResp:
        qtp_dispatcher_stb = dispatchers.QueueToPortDispatcher(
            name, '_stb', configs.numStPorts, configs.numStqEntries, 0, configs.stpAddrW)
        qtp_dispatcher_stb.generate(lsq_submodules, path_rtl)
        lsq_submodules.qtp_dispatcher_stb = qtp_dispatcher_stb

    # Change the name of the following module to lsq_core
    lsq_core = lsq.LSQ(name, '', configs)
    lsq_core.generate(lsq_submodules, path_rtl)
