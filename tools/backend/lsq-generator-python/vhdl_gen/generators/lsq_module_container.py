import vhdl_gen.generators.group_allocator as group_allocator
import vhdl_gen.generators.dispatchers as dispatchers


class GeneratorContainer:
    def __init__(self):
        self.group_allocator: group_allocator.GroupAllocator = None
        self.ptq_dispatcher_lda: dispatchers.PortToQueueDispatcher = None
        self.qtp_dispatcher_ldd: dispatchers.QueueToPortDispatcher = None
        self.ptq_dispatcher_sta: dispatchers.PortToQueueDispatcher = None
        self.ptq_dispatcher_std: dispatchers.PortToQueueDispatcher = None

        # Optional (stResp = True)
        self.qtp_dispatcher_stb: dispatchers.QueueToPortDispatcher = None
