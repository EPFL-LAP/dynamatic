from LSQ.entity import Entity, Architecture, Signal, Signal2D, DeclarativeUnit
from enum import Enum

class ShiftDirection(Enum):
    VERTICAL = 0
    HORIZONATAL = 1

def get_barrel_shifter(
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal2D,
                 output : Signal2D,
                 ):
    declaration = BarrelShifterDecl(parent, unit_name, pointer, to_shift, output)
    return Entity(declaration).get() + Architecture(declaration).get()


class BarrelShifterDecl(DeclarativeUnit):
    def __init__(self, 
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal2D,
                 output : Signal2D,
                 ):
        self.parent = parent
        self.unit_name = unit_name

        self.top_level_comment = ""

        self.entity_port_items = [
            pointer,
            to_shift,
            output
        ]

        assert(pointer.size.number == 1)

        self.local_items = []

        for i in range(pointer.size.bitwidth):
            self.local_items.append(
                ShiftStageSignal(to_shift, i)
                )

        self.body = [
            BarrelShifterBody()
        ]

class BarrelShifterBody():
    def __init__(
            self, 
            pointer : Signal,
            to_shift : Signal2D,
            output : Signal2D
            ):
        self.item = ""

        for i in range(pointer.size.bitwidth):
            self.item += f"""
  -- Check bit {i} if {pointer.base_name}
  -- if 1, shift left by {2^i} 
""".removeprefix
            for j in range(to_shift.size.number):
                self.item += f"""
     {output.base_name}({j}) <= {to_shift.base_name}({(j + 2^i) & to_shift.size.number}); 
""".removeprefix("\n")
            self.item += f"""

""".removeprefix("\n")

    def get(self):
        return self.item

class ShiftStageSignal(Signal2D):
    def __init__(self, to_shift : Signal2D, stage):
        Signal2D.__init__(
            self,
            base_name=f"{to_shift.base_name}_stage_{stage}",
            size=to_shift.size,
            always_number=to_shift.always_number,
            always_vector=to_shift.always_vector
        )


