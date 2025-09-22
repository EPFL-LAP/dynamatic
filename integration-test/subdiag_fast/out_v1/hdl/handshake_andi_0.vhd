-- handshake_andi_0 : andi({'bitwidth': 1, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_andi_0_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_andi_0_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_andi_0_inner_join is
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
architecture arch of handshake_andi_0_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_andi_0_inner_join_and_n port map(ins_valid, allValid);
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

-- Entity of andi
entity handshake_andi_0_inner is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(1 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(1 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(1 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

-- Architecture of andi
architecture arch of handshake_andi_0_inner is
begin
  join_inputs : entity work.handshake_andi_0_inner_join(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => result_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  result <= lhs and rhs;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_andi_0 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    lhs : in std_logic_vector(1 - 1 downto 0);
    lhs_valid : in std_logic;
    lhs_ready : out std_logic;
    lhs_spec : in std_logic_vector(1 - 1 downto 0);
    rhs : in std_logic_vector(1 - 1 downto 0);
    rhs_valid : in std_logic;
    rhs_ready : out std_logic;
    rhs_spec : in std_logic_vector(1 - 1 downto 0);
    result : out std_logic_vector(1 - 1 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic;
    result_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (default)
architecture arch of handshake_andi_0 is
begin
  -- Forward extra signals to output channels
  result_spec <= lhs_spec or rhs_spec;

  inner : entity work.handshake_andi_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      lhs_ready => lhs_ready,
      rhs => rhs,
      rhs_valid => rhs_valid,
      rhs_ready => rhs_ready,
      result => result,
      result_valid => result_valid,
      result_ready => result_ready
    );
end architecture;

