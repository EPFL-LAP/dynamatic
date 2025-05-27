from generators.support.logic import generate_and_n


def generate_join(name, params):
    # Number of input ports
    size = params["size"]

    and_n_module_name = f"{name}_and_n"
    dependencies = generate_and_n(and_n_module_name, {"size": size})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity {name} is
  port (
    -- inputs
    ins_valid  : in std_logic_vector({size} - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

    architecture = f"""
-- Architecture of join
architecture arch of {name} is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.{and_n_module_name} port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector({size} - 1 downto 0);
  begin
    for i in 0 to {size} - 1 loop
      singlePValid(i) := '1';
      for j in 0 to {size} - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to {size} - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;
"""

    return dependencies + entity + architecture
