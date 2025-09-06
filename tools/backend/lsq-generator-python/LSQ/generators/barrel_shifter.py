from LSQ.entity import Entity, Architecture, Signal, Signal2D, DeclarativeUnit
from enum import Enum

class ShiftDirection(Enum):
    VERTICAL = 0
    HORIZONTAL = 1

def get_barrel_shifter(
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal2D,
                 output : Signal2D,
                 direction : ShiftDirection
                 ):
    declaration = BarrelShifterDecl(parent, unit_name, pointer, to_shift, output, direction)
    return Entity(declaration).get() + Architecture(declaration).get()


class BarrelShifterDecl(DeclarativeUnit):
    def __init__(self, 
                 parent : str, 
                 unit_name : str,
                 pointer : Signal,
                 to_shift : Signal2D,
                 output : Signal2D,
                 direction : ShiftDirection
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
            BarrelShifterBody(pointer, to_shift, output, direction)
        ]

class BarrelShifterBody():
    def __init__(
            self, 
            pointer : Signal,
            to_shift : Signal2D,
            output : Signal2D,
            direction : ShiftDirection
            ):
        self.item = ""

        num_stages = pointer.size.bitwidth

        if direction == ShiftDirection.VERTICAL:
            num_shifts = to_shift.size.number
        elif direction == ShiftDirection.HORIZONTAL:
            num_shifts = to_shift.size.bitwidth
            wrapper_size = to_shift.size.number

        shift_ins = [f"{to_shift.base_name}_i"]
        shift_outs = []

        pointer_name = f"{pointer.base_name}_i"

        for i in range(num_stages - 1):
            shift_ins.append(f"stage_{i+1}")
            shift_outs.append(f"stage_{i+1}")

        shift_outs.append(f"{output.base_name}_o")

        for i in range(pointer.size.bitwidth):
            if direction == ShiftDirection.VERTICAL:
                self.item += f"""
  -- Check bit {i} of {pointer.base_name}
  -- if '1', shift left by {(2**i)} 
  shift_stage_{i + 1} : for i in 0 to {num_shifts} - 1 generate

    -- value at 0 in input goes to value at {2**i} in output
    {shift_outs[i]}((i + {2**i}) mod {num_shifts}) <= 
      {shift_ins[i]}(i) when {pointer_name}({i}) = '1' 
        else
      {shift_ins[i]}((i + {2**i}) mod {num_shifts});

  end generate;
""".removeprefix("\n")
            elif direction == ShiftDirection.HORIZONTAL:
                self.item += f"""
  -- Check bit {i} of {pointer.base_name}
  -- if '1', shift left by {(2**i)} 
  shift_stage_{i + 1}_wrapper : for i in 0 to {wrapper_size} - 1 generate
    shift_stage_{i + 1} : for j in 0 to {num_shifts} - 1 generate

      -- value at 0 in input goes to value at {2**i} in output
      {shift_outs[i]}(i)((j + {2**i}) mod {num_shifts}) <= 
        {shift_ins[i]}(i)(j) when {pointer_name}({i}) = '1' 
          else
        {shift_ins[i]}(i)((j + {2**i}) mod {num_shifts});

    end generate;
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
            base_name=f"stage_{stage}",
            size=to_shift.size,
            always_number=to_shift.always_number,
            always_vector=to_shift.always_vector
        )


