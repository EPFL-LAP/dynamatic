from enum import Enum

class InstCxnType(Enum):
    INPUT = "_i"
    OUTPUT = "_o"
    LOCAL = ""

class Signal():
    class Size():
        def __init__(self, bitwidth, number):
            self.bitwidth = bitwidth
            self.number = number

    class Direction(Enum):
        INPUT = 1
        OUTPUT = 2

    def signal_size_to_type_declaration(self):
        if self.size.bitwidth == 1:
            return "std_logic"
        else:
            return f"std_logic_vector({self.size.bitwidth} - 1 downto 0)"

    base_name: str
    direction: Direction
    size: Size

    def __init__(
            self, 
            base_name: str, 
            size : Size, 
            direction : Direction = None
        ):
        self.base_name = base_name
        self.size = size
        self.direction = direction

    def _get_io_suffix(self, name):
        if name == "rst" or name == "clk":
            return ""
        match self.direction:
            case Signal.Direction.INPUT:
                return "_i"
            case Signal.Direction.OUTPUT:
                return "_o"


    def _get_entity_single(self, name):
        match self.direction:
            case Signal.Direction.INPUT:
                # with space at the end to match witdth of out
                direction = "in "
            case Signal.Direction.OUTPUT:
                direction = "out"

        io_suffix = self._get_io_suffix(name)

        type_declaration = self.signal_size_to_type_declaration()

        name = f"{name}{io_suffix}".ljust(30)

        full_declaration = f"{name} : {direction} {type_declaration};"

        # comment out if bitwidth is 0
        if self.size.bitwidth == 0:
            full_declaration = f"-- {full_declaration}"

        return f"""
    {full_declaration}
""".removeprefix("\n")
    
    def _get_local_single(self, name):
        type_declaration = self.signal_size_to_type_declaration()

        name = f"{name}".ljust(35)

        full_declaration = f"signal {name} : {type_declaration};"

        # comment out if bitwidth is 0
        if self.size.bitwidth == 0:
            full_declaration = f"-- {full_declaration}"

        return f"""
  {full_declaration}
""".removeprefix("\n")
    
        
    def _get_inst_single(self, name, cxn_type):
        io_suffix = self._get_io_suffix(name)
        io_name = f"{name}{io_suffix}".ljust(25)

        cxn_name = f"{name}{cxn_type.value}"

        full_declaration = f"{io_name} => {cxn_name},"

        # comment out if bitwidth is 0
        if self.size.bitwidth == 0:
            full_declaration = f"-- {full_declaration}"

        return f"""
      {full_declaration}
""".removeprefix("\n")
    
    def _get_item(self, get_single):
        # if item is singular
        # just generate it using the base name
        if self.size.number == 1:
            return get_single(self.base_name)
        
        # if this item is actually multiple items
        # generate all of them, using indexed names
        all_items = ""
        for i in range(self.size.number):
            item_name = f"{self.base_name}_{i}"
            all_items += get_single(item_name)

        return all_items

    def get_entity_item(self):
        return self._get_item(self._get_entity_single)

    def get_local_item(self):
        return self._get_item(self._get_local_single)

    def get_inst_item(self, cxn_type : InstCxnType):
        def get_single(name) : return self._get_inst_single(name, cxn_type)
        return self._get_item(get_single)
    


class SimpleInstantiation():
    def __init__(self, signal : Signal, cxn_type : InstCxnType):
        self.signal = signal
        self.cxn_type = cxn_type

    def get_inst_item(self):
        return self.signal.get_inst_item(self.cxn_type)

class EntityComment():
    def __init__(self, comment):
        self.comment = comment
    
    def get_entity_item(self):
        return self.comment