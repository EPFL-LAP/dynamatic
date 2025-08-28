from enum import Enum

class Signal():
    class Size():
        def __init__(self, bitwidth, number):
            self.bitwidth = bitwidth
            self.number = number

    class Direction(Enum):
        INPUT = 1
        OUTPUT = 2

    def signalSizeToTypeDeclaration(self):
        if self.size.bitwidth == 1:
            return "std_logic"
        else:
            return f"std_logic_vector({self.size.bitwidth} - 1 downto 0)"

    base_name: str
    direction: Direction
    size: Size

    def __init__(self, base_name: str, direction : Direction, size : Size):
        self.base_name = base_name
        self.direction = direction
        self.size = size


    def _get_entity_single(self, name):
        match self.direction:
            case Signal.Direction.INPUT:
                io_suffix = "i"
                # with space at the end to match witdth of out
                direction = "in "
            case Signal.Direction.OUTPUT:
                io_suffix = "o"
                direction = "out"

        type_declaration = self.signalSizeToTypeDeclaration()

        name = f"{name}_{io_suffix}".ljust(30)

        full_declaration = f"{name} : {direction} {type_declaration};"

        # comment out if bitwidth is 0
        if self.size.bitwidth == 0:
            full_declaration = f"-- {full_declaration}"

        return f"""
    {full_declaration}
""".removeprefix("\n")

    def get_entity_signal(self):
        # if item is singular
        # just generate it using the base name
        if self.size.number == 1:
            return self._get_entity_single(self.base_name)
        
        # if this item is actually multiple items
        # generate all of them, using indexed names
        all_items = ""
        for i in range(self.size.number):
            item_name = f"{self.base_name}_{i}"
            all_items += self._get_entity_single(item_name)

        return all_items

    def get_local_signal(self):
        # if item is singular
        # just generate it using the base name
        if self.size.number == 1:
            return self._get_entity_single(self.base_name)
        
        # if this item is actually multiple items
        # generate all of them, using indexed names
        all_items = ""
        for i in range(self.size.number):
            item_name = f"{self.base_name}_{i}"
            all_items += self._get_entity_single(item_name)

        return all_items


class EntityComment():
    def __init__(self, comment):
        self.comment = comment
    
    def get(self):
        return self.comment