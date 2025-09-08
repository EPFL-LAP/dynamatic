
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.entity import Signal, Signal2D, Instantiation, SimpleInstantiation, InstCxnType, DeclarativeUnit, Entity, Architecture, RTLComment
import LSQ.declarative_signals as ds

from collections import defaultdict

from LSQ.generators.barrel_shifter import get_barrel_shifter_1D, ShiftDirection

def get_write_enables(config, queue_type : QueueType, parent_name):
    declaration = WriteEnablesDecl(config, parent_name, queue_type)
    unit = Entity(declaration).get() + Architecture(declaration).get()

    barrel_shifters = _get_barrel_shifter(config, declaration, queue_type)

    return barrel_shifters + unit



class WriteEnablesDecl(DeclarativeUnit):
    def __init__(self, config: Config, parent_name, queue_type : QueueType):
        self.top_level_comment = f"""
-- {queue_type.value.capitalize()} Queue Write Enables Unit
-- Sub-unit of the Group Allocator.
--
-- Generates the write enable signals for the {queue_type.value} queue
-- based on the number of {queue_type.value} queue entries being allocated
-- and the tail pointers of the {queue_type.value} queue.
--
-- First, the number of write enable signals to set high is decided
-- based on the "number of new entries to the {queue_type.value} queue".
-- 
-- This "number of new entries to the {queue_type.value} queue" is also used
-- by the {queue_type.value} queue itself, to update its tail pointer.
--
-- Then the write enables are shifted into the correct place 
-- for the internal circular buffer,
-- based on the {queue_type.value} tail pointer.
""".strip()
        
        self.initialize_name(
            parent_name=parent_name,
            unit_name=WRITE_ENABLE_NAME(queue_type)
        )


        d = Signal.Direction
        self.entity_port_items = [
            RTLComment(
                f"""

    -- Input: Number of New Queue Entries to Allocate (N)
    -- The first N write enables signals are set to high.
"""
            ),
            ds.NumNewEntries(
                config, 
                queue_type, 
                d.INPUT
            ),

            RTLComment(
                f"""
    -- Input: {queue_type.value} queue pointer
    -- Used to shift the write enables into the correct alignment

"""
            ),
            ds.QueuePointer(
                config, 
                queue_type, 
                QueuePointerType.TAIL,
                d.INPUT
            ),


            RTLComment(
                f"""

    -- Output: Shifted write enable signals

"""
            ),
            ds.QueueWriteEnable(
                config, 
                queue_type,
                d.OUTPUT
            )
        ]

        self.local_items = [
            WriteEnable(config, queue_type, shifted=False),
            WriteEnable(config, queue_type, shifted=True)
        ]

        self.body = [
            WriteEnablesUnshifted(config, queue_type),
            BarrelShiftInstantiation(config, self.name(), queue_type),
            OutputAssignments(config, queue_type)
        ]

class WriteEnablesUnshifted():
    def __init__(self, config : Config, queue_type : QueueType):
        bitwidth = config.queue_idx_bitwidth(queue_type)

        new_entries = NUM_NEW_ENTRIES_NAME(queue_type)

        unshf_wen = UNSHIFTED_WRITE_ENABLE_NAME(queue_type)

        self.item = f"""
  -- For each write enable, if its index is less than the number of new entries
  -- Set it to 1
  unshifted_write_enables : for i_int in 0 to {config.queue_num_entries(queue_type)} - 1 generate

    -- convert integer for loop iterator
    -- to constant unsigned value in each generated assignment
    constant i : 
      unsigned({bitwidth} - 1 downto 0) := to_unsigned(i_int, {bitwidth});

  begin

    {unshf_wen}(i_int) <= '1' when i < unsigned({new_entries}_i) else '0';

  end generate;

""".removeprefix("\n")
        
    def get(self):
        return self.item

class OutputAssignments():
    def __init__(self, config : Config, queue_type : QueueType):
        self.item = ""
        for i in range(config.queue_num_entries(queue_type)):
            assign_to = f"{WRITE_ENABLE_NAME(queue_type)}_{i}_o"

            if i < 10:
                assign_to += " "

            self.item += f"""
  {assign_to} <= {WRITE_ENABLE_NAME(queue_type)}({i});
""".removeprefix("\n")
            
    def get(self):
        return self.item

class WriteEnable(Signal):
    """
    Bitwidth = N
    Number = 1

    Single vector storing the 
    (unshifted/shifted) write enable per queue entry
        
    Bitwidth is equal to the number of queue entries
    Number is 1
    """
    def __init__(
            self, 
            config : Config,
            queue_type : QueueType,
            shifted = False,
            # input/output of the barrel shifters
            direction : Signal.Direction = None
            ):

        if shifted:
            base_name = WRITE_ENABLE_NAME(queue_type)
        else:
            base_name = UNSHIFTED_WRITE_ENABLE_NAME(queue_type)

        Signal2D.__init__(
            self,
            base_name=base_name,
            direction=direction,
            size=Signal.Size(
                bitwidth=config.queue_num_entries(queue_type),
                number=1
            )
        )


# Barrel shifter aligns write enables to queue
class BarrelShiftInstantiation(Instantiation):
    def __init__(self, config : Config, parent, queue_type : QueueType):
        si = SimpleInstantiation
        d = Signal.Direction
        c = InstCxnType

        # Barrel shifters are generated uniquely
        # per use
        #
        # and so their input and output signals
        # match the signals pass to the instantiation
        port_items = [
            # shift the store orders horizontally
            # based on the store queue tail
            si(
                ds.QueuePointer(
                    config, 
                    queue_type,
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                c.INPUT
            ),
            # pass in the unshifted write enables
            si(
                WriteEnable(
                    config,
                    queue_type,
                    shifted=False,
                    direction = d.INPUT
                ),
                c.LOCAL
            ),
            # pass out the write enables
            # shifted for the queue
            si(
                WriteEnable(
                    config,
                    queue_type,
                    shifted=True,
                    direction = d.OUTPUT
                ),
                c.LOCAL
            )
        ]

        Instantiation.__init__(
            self,
            "barrel_shift",
            parent,
            port_items,
            comment=f"""
  -- Shift the bits representing the write enables
  -- Based on the {queue_type.value} queue tail pointer
  -- So that bit 0 moves to bit (tail pointer)
  -- Making the order aligned with the {queue_type.value} queue
""".strip()
        )

def _get_barrel_shifter(config, declaration, queue_type : QueueType):

    d = Signal.Direction

    
    return get_barrel_shifter_1D(
        declaration.name(),
        "barrel_shift",
        ds.QueuePointer(
            config, 
            queue_type,
            QueuePointerType.TAIL,
            d.INPUT
        ),
        WriteEnable(
            config,
            queue_type,
            shifted=False,
            direction = d.INPUT
        ),
        WriteEnable(
            config,
            queue_type,
            shifted=True,
            direction = d.OUTPUT
        ),
        comment=f"""
-- Barrel shifter for the write enables
-- Aligns write enable bits with {queue_type.value} queue
""".strip()
    )