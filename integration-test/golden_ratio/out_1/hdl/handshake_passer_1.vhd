-- handshake_passer_1 : passer({'bitwidth': 1})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_passer_1_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_passer_1_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_passer_1_inner_join is
  port (
    -- inputs
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join
architecture arch of handshake_passer_1_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_passer_1_inner_join_and_n port map(ins_valid, allValid);
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

-- Entity of passer_dataless
entity handshake_passer_1_inner is
  port (
    clk, rst : in std_logic;
    data_valid : in std_logic;
    data_ready : out std_logic;
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    result_valid : out std_logic;
    result_ready : in std_logic
  );
end entity;

-- Architecture of passer_dataless
architecture arch of handshake_passer_1_inner is
  signal branch_valid, branch_ready : std_logic;
begin
  branch_ready <= not ctrl(0) or result_ready;
  result_valid <= branch_valid and ctrl(0);
  join_inputs : entity work.handshake_passer_1_inner_join(arch)
    port map(
      -- inputs
      ins_valid(0) => data_valid,
      ins_valid(1) => ctrl_valid,
      outs_ready   => branch_ready,
      -- outputs
      outs_valid   => branch_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => ctrl_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of passer
entity handshake_passer_1 is
  port (
    clk, rst : in std_logic;
    data : in std_logic_vector(1 - 1 downto 0);
    data_valid : in std_logic;
    data_ready : out std_logic;
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    result : out std_logic_vector(1 - 1 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic
  );
end entity;

-- Architecture of passer
architecture arch of handshake_passer_1 is
begin
  inner : entity work.handshake_passer_1_inner(arch)
    port map(
      clk          => clk,
      rst          => rst,
      data_valid   => data_valid,
      data_ready   => data_ready,
      ctrl         => ctrl,
      ctrl_valid   => ctrl_valid,
      ctrl_ready   => ctrl_ready,
      result_valid => result_valid,
      result_ready => result_ready
    );

  result <= data;
end architecture;

