import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers


class LSQ_Submodules:
    """
    Save LSQ (Load-Store Queue) submodule instances.

    This class acts as a simple struct to group together all the generator objects 
    required to build a complete LSQ. An instance of this class is created by the 
    codegen.py script, and then passed to the LSQ generator.

    Attributes:
        group_allocator (group_allocator.GroupAllocator):
            The generator instance for the Group Allocator module.
        ptq_dispatcher_lda (dispatchers.PortToQueueDispatcher):
            The Port-to-Queue dispatcher for the Load Address (LDA) channel.
        qtp_dispatcher_ldd (dispatchers.QueueToPortDispatcher):
            The Queue-to-Port dispatcher for the Load Data (LDD) channel.
        ptq_dispatcher_sta (dispatchers.PortToQueueDispatcher):
            The Port-to-Queue dispatcher for the Store Address (STA) channel.
        ptq_dispatcher_std (dispatchers.PortToQueueDispatcher):
            The Port-to-Queue dispatcher for the Store Data (STD) channel.
        qtp_dispatcher_stb (dispatchers.QueueToPortDispatcher):
            The optional Queue-to-Port dispatcher for the Store Backward/Response
            (STB) channel. This is only instantiated if store responses are
            enabled in the configuration.
    """

    def __init__(self):
        self.group_allocator: group_allocator.GroupAllocator = None
        self.ptq_dispatcher_lda: dispatchers.PortToQueueDispatcher = None
        self.qtp_dispatcher_ldd: dispatchers.QueueToPortDispatcher = None
        self.ptq_dispatcher_sta: dispatchers.PortToQueueDispatcher = None
        self.ptq_dispatcher_std: dispatchers.PortToQueueDispatcher = None

        # Optional (stResp = True)
        self.qtp_dispatcher_stb: dispatchers.QueueToPortDispatcher = None
