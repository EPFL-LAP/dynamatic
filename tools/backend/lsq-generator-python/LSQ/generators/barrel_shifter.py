from LSQ.entity import Entity, Architecture, Signal, DeclarativeUnit
from enum import Enum

class ShiftDirection(Enum):
    VERTICAL = 0
    HORIZONATAL = 1

def get_barrel_shifter(
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal,
                 output : Signal,
                 ):
    declaration = BarrelShifterDecl(parent, unit_name, pointer, to_shift, output)
    return Entity(declaration).get() + Architecture(declaration).get()


class BarrelShifterDecl(DeclarativeUnit):
    def __init__(self, 
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal,
                 output : Signal,
                 ):
        self.parent = parent
        self.unit_name = unit_name

        self.entity_port_items = [
            pointer,
            to_shift,
            output
        ]

        assert(pointer.size.number == 1)
        self.num_stages = pointer.size.bitwidth

        self.local_items = []

        for i in range(self.num_stages):
            self.local_items.append(
                ShiftStageSignal(to_shift, i)
                )

        self.body = [

        ]

class BarrelShifterBody():
    def __init__(
            self, 
            pointer : Signal,
            to_shift : Signal,
            output : Signal
            ):
        pass

    def get(self):
        return ""

class ShiftStageSignal(Signal):
    def __init__(self, to_shift : Signal, stage):
        Signal.__init__(
            self,
            base_name=f"{to_shift.base_name}_stage_{stage}",
            size=to_shift.size,
            always_number=to_shift.always_number,
            always_vector=to_shift.always_vector
        )


