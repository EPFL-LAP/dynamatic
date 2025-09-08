from LSQ.entity import Signal, RTLComment, Instantiation, SimpleInstantiation, InstCxnType, Signal2D
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.utils import bin_string, one_hot, mask_until

import LSQ.declarative_signals as ds


class GroupAllocatorBodyItems():
    class GroupHandshakingInst(Instantiation):
        def __init__(self, config : Config, parent):

            c = InstCxnType

            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitValid(
                        config
                    ), 
                    c.INPUT
                ),
                si(
                    ds.GroupInitReady(
                        config,
                    ), 
                    c.OUTPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.LOAD, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                ),
                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.LOAD, 
                        QueuePointerType.HEAD,
                        d.INPUT
                        ), 
                    c.INPUT
                ),
                si(
                    ds.QueueIsEmpty(
                        QueueType.LOAD,
                        d.INPUT
                    ), 
                c.INPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.STORE, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.STORE, 
                        QueuePointerType.HEAD,
                        d.INPUT
                    ), 
                    c.INPUT
                ),
                
                si(
                    ds.QueueIsEmpty(
                        QueueType.STORE,
                        d.INPUT
                    ), 
                    c.INPUT
                ),



                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.OUTPUT
                    ), 
                    c.LOCAL
                )
            ]


            Instantiation.__init__(
                self,
                unit_name=GROUP_HANDSHAKING_NAME,
                parent=parent,
                port_items=port_items
            )

    class PortIdxPerEntryInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, parent):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.QueuePointer(
                    config, 
                    queue_type, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.PortIdxPerEntry(
                    config, 
                    queue_type,
                    d.OUTPUT
                    ),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                unit_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                parent=parent,
                port_items=port_items
            )

    class NaiveStoreOrderPerEntryInst(Instantiation):
        def __init__(self, config : Config, parent):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ds.GroupInitTransfer(
                    config, 
                    d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(ds.QueuePointer(
                    config, 
                    QueueType.LOAD, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.QueuePointer(
                    config, 
                    QueueType.STORE, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.NaiveStoreOrderPerEntry(
                    config,
                    d.OUTPUT
                    ),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                unit_name=NAIVE_STORE_ORDER_PER_ENTRY_NAME,
                parent=parent,
                port_items=port_items
            )

    class NumNewQueueEntriesInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, parent):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.NumNewQueueEntries(
                        config, 
                        queue_type,
                        d.OUTPUT
                    ),
                    c.LOCAL
                )
            ]

            Instantiation.__init__(
                self,
                unit_name=NUM_NEW_ENTRIES_NAME(queue_type),
                parent=parent,
                port_items=port_items
            )

    class WriteEnableInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, parent):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation

            port_items = [
                si(
                    ds.NumNewQueueEntries(
                        config, 
                        queue_type, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        queue_type, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                    ),
                si(
                    ds.QueueWriteEnable(
                        config, 
                        queue_type,
                        d.OUTPUT
                    ),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                unit_name=WRITE_ENABLE_NAME(queue_type),
                parent=parent,
                port_items=port_items
            )
    
    class NumNewEntriesAssignment():
        def get(self):
            return f"""
    -- the "number of new entries" signals are local, 
    -- since they are used to generate the write enable signals
    --
    -- Here we drive the outputs with them
    {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)};
    {NUM_NEW_ENTRIES_NAME(QueueType.STORE)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.STORE)};

""".removeprefix("\n")
                    