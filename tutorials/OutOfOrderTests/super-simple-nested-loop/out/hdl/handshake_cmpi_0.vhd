-- handshake_cmpi_0 : cmpi({'port_types': {'lhs': '!handshake.channel<i7>', 'rhs': '!handshake.channel<i7>', 'result': '!handshake.channel<i1>'}, 'predicate': 'ult', 'bitwidth': 7, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_cmpi_0_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_cmpi_0_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join_dataless
entity handshake_cmpi_0_join is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join_dataless
architecture arch of handshake_cmpi_0_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_cmpi_0_join_and_n port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(2 - 1 downto 0);
  begin
    for i in 0 to 2 - 1 loop
      singlePValid(i) := '1';
      for j in 0 to 2 - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to 2 - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpi
entity handshake_cmpi_0 is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(7 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(7 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

-- Architecture of cmpi
architecture arch of handshake_cmpi_0 is
begin
  join_inputs : entity work.handshake_cmpi_0_join(arch)
    port map(
      clk          => clk,
      rst          => rst,
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => result_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  result(0) <= '1' when (unsigned(lhs) < unsigned(rhs)) else '0';
end architecture;

