from typing import List


def generate_ram(name, params):
    data_width = params["data_width"]
    addr_width = params["addr_width"]
    size = params["size"]
    values = params["values"]
    return _generate_ram(
        name,
        data_width,
        addr_width,
        size,
        values,
    )


def _generate_ram(
    name: str,
    data_width: int,
    addr_width: int,
    size: int,
    values: List[int],
):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
  port (
    clk       : in std_logic;
    rst       : in std_logic;
    -- from circuit (mem_controller / LSQ)
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector({addr_width} - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector({addr_width} - 1 downto 0);
    storeData : in std_logic_vector({data_width} - 1 downto 0);
    -- to circuit (mem_controller / LSQ)
    loadData  : out std_logic_vector({data_width} - 1 downto 0)
  );
end entity;
    """

    architecture = f"""
architecture arch of {name} is
  type ram_type is array (0 to {size} - 1) of std_logic_vector({data_width} - 1 downto 0);
  {_gen_intial_block(data_width, size, values)}
begin
  read_proc : process(clk)
  begin
    if (rising_edge(clk)) then
      if (loadEn = '1') then
        loadData <= ram(to_integer(unsigned(loadAddr)));
      end if;
    end if;
  end process;

  write_proc : process(clk)
  begin
    if (rising_edge(clk)) then
      if (storeEn = '1') then
        ram(to_integer(unsigned(storeAddr))) <= storeData;
      end if;
    end if;
  end process;
end architecture;
    """

    return entity + architecture


"""
Returns the 2's complement binary representation of integer `n` with
the given `bitwidth`.
"""


def _to_twos_complement(n, bitwidth, addr):
    if n < 0:
        n = (1 << bitwidth) + n
    if n >= (1 << bitwidth) or n < 0:
        raise ValueError(
            f"""
            The memory cannot be correctly instantiated, since the value
            {n} at address {addr} doesn't fit in {bitwidth} bits.
            """
        )
    return format(n, f"0{bitwidth}b")


def _gen_intial_block(data_width: int, size: int, init_vals: List[int]):

    if init_vals == []:
        return "  signal ram : ram_type;\n"

    init_strings = []

    for addr, val in enumerate(init_vals):
        init_strings.append(
            '"' + _to_twos_complement(val, data_width, addr) + '"')

    if len(init_vals) < int(size):
        for _ in range(int(size) - len(init_vals)):
            init_strings.append('"' + f"{0:0{data_width}b}" + '"')

    return "signal ram : ram_type := (" + ",\n".join(init_strings)  + ");"
