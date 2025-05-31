#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# This file redesigned the classes to represent the signals in the VHDL file
import re
import math

# ===----------------------------------------------------------------------===#
# VHDL Signal Type Definition
# ===----------------------------------------------------------------------===#

#
#   std_logic: a single bit
#


class VHDLLogicType:
    """The functionality of this class is similar to Class Logic
    Instead of storing the string in a global variable,
    this class now directly return the generated string
    """

    def __init__(self, name: str, type: str = "w"):
        # Type must be one of the following 4 letters
        assert type in ("i", "o", "w", "r")
        self.name = name
        self.type = type

    def __repr__(self) -> str:
        # Signal type
        type = ""
        if self.type == "w":
            type = "wire"
        elif self.type == "i":
            type = "input"
        elif self.type == "o":
            type = "output"
        elif self.type == "r":
            type = "reg"
        return f"name: {self.name}\n" + f"type: {type}\n" + f"size: single bit\n"

    def getNameRead(self, suffix="") -> str:
        if self.type == "w":
            return self.name + suffix
        elif self.type == "r":
            return self.name + suffix + "_q"
        elif self.type == "i":
            return self.name + suffix
        elif self.type == "o":
            raise TypeError(f'Cannot read from the output signal "{self.name}"!')

    def getNameWrite(self, suffix="") -> str:
        if self.type == "w":
            return self.name + suffix
        elif self.type == "r":
            return self.name + suffix + "_d"
        elif self.type == "i":
            raise TypeError(f'Cannot write to the input signal "{self.name}"!')
        elif self.type == "o":
            return self.name + suffix

    def signalInit(self, suffix: str = ""):
        signal_str = ""

        # Change the name of the signal to match the naming convention
        # in Dynamatic
        if suffix != "":
            name_list = self.name.split("_")

            if suffix == "0":
                new_name_list = name_list[:-1] + [suffix, name_list[-1]]
            else:
                new_name_list = name_list[:-2] + [suffix, name_list[-1]]

            self.name = "_".join(new_name_list)

        # Check the signal type
        if self.type == "w":
            signal_str += f"\tsignal {self.name} : std_logic;\n"
        elif self.type == "r":
            signal_str += f"\tsignal {self.name}_d : std_logic;\n"
            signal_str += f"\tsignal {self.name}_q : std_logic;\n"
        elif self.type == "i":
            signal_str += ";\n"
            signal_str += f"\t\t{self.name} : in std_logic"
        elif self.type == "o":
            signal_str += ";\n"
            signal_str += f"\t\t{self.name} : out std_logic"

        return signal_str

    def regInit(self, enable=None, init=None):
        assert self.type == "r"

        reg_init_str = ""

        # Process Content init
        if init != None:
            reg_init_str += "\t\tif (rst = '1') then\n"
            reg_init_str += f"\t\t\t{self.getNameRead()} <= {IntToBits(init)};\n"
            reg_init_str += "\t\telsif (rising_edge(clk)) then\n"
        else:
            reg_init_str += "\t\tif (rising_edge(clk)) then\n"

        if enable != None:
            reg_init_str += f"\t\t\tif ({enable.getNameRead()} = '1') then\n"
            reg_init_str += f"\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n"
            reg_init_str += "\t\t\tend if;\n"
        else:
            reg_init_str += f"\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n"

        reg_init_str += "\t\tend if;\n"

        return reg_init_str


#
#   std_logic_vec
#


class VHDLLogicVecType(VHDLLogicType):
    """The functionality of this class is similar to Class LogicVec
    Instead of storing the string in a global variable,
    this class now directly return the generated string

    The port initialization is removed from the class init function
    """

    def __init__(self, name: str, type: str = "w", size: int = 1):
        VHDLLogicType.__init__(self, name, type)

        # The size must be larger than 0
        assert size > 0
        self.size = size

    def __repr__(self) -> str:
        # Signal type
        type = ""
        if self.type == "w":
            type = "wire"
        elif self.type == "i":
            type = "input"
        elif self.type == "o":
            type = "output"
        elif self.type == "r":
            type = "reg"
        return f"name: {self.name}\n" + f"type: {type}\n" + f"size: {self.size}\n"

    def getNameRead(self, i=None, suffix="") -> str:
        if i == None:
            return VHDLLogicType.getNameRead(self, suffix)
        else:
            assert i < self.size
            return VHDLLogicType.getNameRead(self, suffix) + f"({i})"

    def getNameWrite(self, i=None, suffix="") -> str:
        if i == None:
            return VHDLLogicType.getNameWrite(self, suffix)
        else:
            assert i < self.size
            return VHDLLogicType.getNameWrite(self, suffix) + f"({i})"

    def signalInit(self, suffix=""):
        signal_str = ""

        # Change the name of the signal to match the naming convention
        # in Dynamatic
        if suffix != "":
            name_list = self.name.split("_")
            if suffix == "0":
                new_name_list = name_list[:-1] + [suffix, name_list[-1]]
            else:
                new_name_list = name_list[:-2] + [suffix, name_list[-1]]

            self.name = "_".join(new_name_list)

        if self.type == "w":
            signal_str += (
                f"\tsignal {self.name} : std_logic_vector({self.size - 1} downto 0);\n"
            )
        elif self.type == "r":
            signal_str += (
                f"\tsignal {self.name}_d : std_logic_vector({self.size - 1} downto 0);\n"
            )
            signal_str += (
                f"\tsignal {self.name}_q : std_logic_vector({self.size - 1} downto 0);\n"
            )
        elif self.type == "i":
            # For the wrapper, we don't add i/o in the port name
            signal_str += ";\n"
            signal_str += (
                f"\t\t{self.name} : in std_logic_vector({self.size - 1} downto 0)"
            )
        elif self.type == "o":
            # For the wrapper, we don't add i/o in the port name
            signal_str += ";\n"
            signal_str += (
                f"\t\t{self.name} : out std_logic_vector({self.size - 1} downto 0)"
            )

        return signal_str

    def regInit(self, enable=None, init=None):
        reg_str = ""
        assert self.type == "r"
        if init != None:
            reg_str += "\t\tif (rst = '1') then\n"
            reg_str += f"\t\t\t{self.getNameRead()} <= {IntToBits(init, self.size)};\n"
            reg_str += "\t\telsif (rising_edge(clk)) then\n"
        else:
            reg_str += "\t\tif (rising_edge(clk)) then\n"

        if enable != None:
            reg_str += f"\t\t\tif ({enable.getNameRead()} = '1') then\n"
            reg_str += f"\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n"
            reg_str += "\t\t\tend if;\n"
        else:
            reg_str += f"\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n"
        reg_init_str += "\t\tend if;\n"

        return reg_str


#
#   An array of std_logic
#


class VHDLLogicTypeArray(VHDLLogicType):
    """An array of std_logic"""

    def __init__(self, name: str, type: str = "w", length: int = 1):
        VHDLLogicType.__init__(self, name, type)
        self.length = length

    def __repr__(self) -> str:
        return VHDLLogicType.__repr__(self) + f"array length: {self.length}"

    def getNameRead(self, i) -> str:
        assert i in range(0, self.length)
        return VHDLLogicType.getNameRead(self)

    def getNameWrite(self, i) -> str:
        assert i in range(0, self.length)
        return VHDLLogicType.getNameWrite(self)

    def signalInit(self):
        """We return all the definitions as a string"""
        signals_str = ""

        for i in range(0, self.length):
            signals_str += VHDLLogicType.signalInit(self, f"{i}")

        return signals_str

    def __getitem__(self, i) -> VHDLLogicType:
        assert i in range(0, self.length)
        name = re.sub(
            r"(\D+)\d+(\D+)", lambda m: f"{m.group(1)}{i}{m.group(2)}", self.name
        )
        return VHDLLogicType(name, self.type)

    def regInit(self, enable=None, init=None):
        assert self.type == "r"

        # Define the output string
        reg_init_str = ""

        if init != None:
            reg_init_str += "\t\tif (rst = '1') then\n"
            for i in range(0, self.length):
                reg_init_str += (
                    f"\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i])};\n"
                )
            reg_init_str += "\t\telsif (rising_edge(clk)) then\n"
        else:
            reg_init_str += "\t\tif (rising_edge(clk)) then\n"

        if enable != None:
            for i in range(0, self.length):
                reg_init_str += f"\t\t\tif ({enable.getNameRead(i)} = '1') then\n"
                reg_init_str += (
                    f"\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n"
                )
                reg_init_str += "\t\t\tend if;\n"
        else:
            for i in range(0, self.length):
                reg_init_str += (
                    f"\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n"
                )

        reg_init_str += "\t\tend if;\n"

        return reg_init_str


#
#   An array of std_logic vector
#


class VHDLLogicVecTypeArray(VHDLLogicVecType):
    """An array of std_logic vector"""

    def __init__(self, name: str, type: str = "w", length: int = 1, size: int = 1):
        self.length = length
        VHDLLogicVecType.__init__(self, name, type, size)

    def __repr__(self) -> str:
        return VHDLLogicVecType.__repr__(self) + f"array length: {self.length}"

    def getNameRead(self, i, j=None) -> str:
        assert i in range(0, self.length)
        return VHDLLogicVecType.getNameRead(self, i)

    def getNameWrite(self, i, j=None) -> str:
        assert i in range(0, self.length)
        return VHDLLogicVecType.getNameWrite(self, j)

    def __getitem__(self, i) -> VHDLLogicVecType:
        assert i in range(0, self.length)
        name = re.sub(
            r"(\D+)\d+(\D+)", lambda m: f"{m.group(1)}{i}{m.group(2)}", self.name
        )
        return VHDLLogicVecType(name, self.type, self.size)

    def signalInit(self):
        sig_init_str = ""

        for i in range(0, self.length):
            sig_init_str += VHDLLogicVecType.signalInit(self, f"{i}")

        return sig_init_str

    def regInit(self, enable=None, init=None):
        # Define the output string
        reg_init_str = ""

        assert self.type == "r"

        if init != None:
            reg_init_str += "\t\tif (rst = '1') then\n"
            for i in range(0, self.length):
                reg_init_str += (
                    f"\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i], self.size)};\n"
                )
            reg_init_str += "\t\telsif (rising_edge(clk)) then\n"
        else:
            reg_init_str += "\t\tif (rising_edge(clk)) then\n"

        if enable != None:
            for i in range(0, self.length):
                reg_init_str += f"\t\t\tif ({enable.getNameRead(i)} = '1') then\n"
                reg_init_str += (
                    f"\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n"
                )
                reg_init_str += "\t\t\tend if;\n"
        else:
            for i in range(0, self.length):
                reg_init_str += (
                    f"\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n"
                )

        reg_init_str += "\t\tend if;\n"

        return reg_init_str


def OpTab(out, tabLevel, *list_in) -> str:
    if type(out) == tuple:
        if len(out) == 2:
            str_ret = "\t" * tabLevel + f"{out[0].getNameWrite(out[1])} <="
        else:
            str_ret = "\t" * tabLevel + f"{out[0].getNameWrite(out[1], out[2])} <="
    else:
        str_ret = "\t" * tabLevel + f"{out.getNameWrite()} <="
        if type(out) == VHDLLogicType:
            size = 1
        else:
            size = out.size
    for arg in list_in:
        if type(arg) == str:
            str_ret += " " + arg
        elif type(arg) == int:
            str_ret += " " + IntToBits(arg, size)
        elif type(arg) == tuple:
            if type(arg[0]) == int:
                str_ret += " " + IntToBits(arg[0], arg[1])
            elif len(arg) == 2:
                str_ret += " " + arg[0].getNameRead(arg[1])
            else:
                str_ret += " " + arg[0].getNameRead(arg[1], arg[2])
        else:
            str_ret += " " + arg.getNameRead()
    str_ret += ";\n"
    return str_ret


# ===----------------------------------------------------------------------===#
# Helper Function
# ===----------------------------------------------------------------------===#


def MaskLess(din, size) -> str:
    if (din > size):
        raise ValueError("Unknown value!")
    return '\"' + '0'*(size-din) + '1'*din + '\"'


def IntToBits(din, size=None) -> str:
    if size == None:
        if din:
            return "'1'"
        else:
            return "'0'"
    else:
        str_ret = '"'
        for i in range(0, size):
            if din % 2 == 0:
                str_ret = "0" + str_ret
            else:
                str_ret = "1" + str_ret
            din = din // 2
        str_ret = '"' + str_ret
        if din != 0:
            raise ValueError("Unknown value!")
        return str_ret


def Zero(size) -> str:
    if size == None:
        return "'0'"
    else:
        return '"' + "0" * size + '"'


def GetValue(row, i) -> int:
    if len(row) > i:
        return row[i]
    else:
        return 0


def isPow2(value: int) -> bool:
    return (value & (value-1) == 0) and value != 0


def log2Ceil(value: int) -> int:
    return math.ceil(math.log2(value))
