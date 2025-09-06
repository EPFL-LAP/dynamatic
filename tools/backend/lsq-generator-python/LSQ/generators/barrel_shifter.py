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

        for i in range(1, pointer.size.bitwidth):
            self.local_items.append(
                ShiftStageSignal(output, i)
                )

        self.body = [
            BarrelShifterBody(pointer, to_shift, output, self.local_items)
        ]

class BarrelShifterBody():
    def __init__(
            self, 
            pointer : Signal,
            to_shift : Signal2D,
            output : Signal2D,
            local_items
            ):
        self.item = ""

        num_stages = pointer.size.bitwidth

        num_shifts = to_shift.size.number

        shift_ins = [to_shift.base_name]
        shift_outs = []

        for i in range(num_stages - 1):
            shift_ins.append(local_items[i].base_name)
            shift_outs.append(local_items[i].base_name)

        shift_outs.append(output.base_name)

        for i in range(pointer.size.bitwidth):
            self.item += f"""
  -- Check bit {i} of {pointer.base_name}
  -- if '1', shift left by {(2**i)} 
  for i in 0 to {num_shifts} - 1 generate
    {shift_outs[i]}((i + {2**i}) mod {num_shifts}) <=
      {shift_ins[i]}(i)
        when std_tail{i} = '1' else
      {shift_ins[i]}((i + {2**i}) mod {num_shifts})

  end generate;
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


