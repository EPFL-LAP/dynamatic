"""
Declarative signal definitions, 
defining their names, numbers, and bitwidths in a single place.
Also provides docstrings for each signal to help remember what it does.
"""

from LSQ.entity import Signal
from LSQ.config import Config
from LSQ.utils import QueueType, QueuePointerType

from LSQ.rtl_signal_names import *


class Reset(Signal):
    """
    Input.

    Generic RTL reset signal

    Bitwidth=1, Number=1
    """
    def __init__(self):
        Signal.__init__(
            self,
            base_name="rst",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=1
            )
        )


class Clock(Signal):
    """
    Input.

    Generic RTL clock signal

    Bitwidth=1, Number=1
    """
    def __init__(self):
        Signal.__init__(
            self,
            base_name="clk",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=1
            )
        )

class GroupInitValid(Signal):
    """
    Input

    Bitwidth = 1, Number = N

    1-bit valid signals for the "group init" channels, from the dataflow circuit. 
    For N groups, there are N "group init" channels, which results in 

    group_init_valid_0_i : in std_logic;
    group_init_ready_1_i : in std_logic;
    .
    .
    .
    group_init_ready_N_i : in std_logic;
    """
    def __init__(self, config : Config):
        Signal.__init__(
            self,
            base_name=f"{GROUP_INIT_CHANNEL_NAME}_valid",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=config.num_groups()
            ),
            always_number=True
        )
    
    @staticmethod
    def comment(config: Config) -> str:
        return f"""

""".removeprefix("\n").removesuffix("\n")
    

class GroupInitReady(Signal):
    """
    Output.
        
    Bitwidth = 1, Number = N

    1-bit ready signals for the "group init" channels, from the dataflow circuit. 
    For N groups, there are N "group init" channels, which results in

    group_init_ready_0_i : out std_logic;
    group_init_ready_1_i : out std_logic;
    .
    .
    .
    group_init_ready_N_i : out std_logic;
    """
    def __init__(self, config : Config):
        Signal.__init__(
            self,
            base_name=f"{GROUP_INIT_CHANNEL_NAME}_ready",
            direction=Signal.Direction.OUTPUT,
            size=Signal.Size(
                bitwidth=1,
                number=config.num_groups()
            ),
            always_number=True
        )