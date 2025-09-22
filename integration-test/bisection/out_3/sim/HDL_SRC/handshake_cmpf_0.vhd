-- handshake_cmpf_0 : cmpf({'is_double': False, 'extra_signals': {}, 'predicate': 'olt'})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_cmpf_0_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_cmpf_0_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_cmpf_0_inner_join is
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
architecture arch of handshake_cmpf_0_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_cmpf_0_inner_join_and_n port map(ins_valid, allValid);
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

-- Entity of cmpf_single_precision
entity handshake_cmpf_0_inner is
  port(
    -- inputs
    clk: in std_logic;
    rst: in std_logic;
    lhs: in std_logic_vector(32 - 1 downto 0);
    lhs_valid: in std_logic;
    rhs: in std_logic_vector(32 - 1 downto 0);
    rhs_valid: in std_logic;
    result_ready: in std_logic;
    -- outputs
    unordered: out std_logic;
    XltY: out std_logic;
    XeqY: out std_logic;
    XgtY: out std_logic;
    XleY: out std_logic;
    XgeY: out std_logic;
    result_valid: out std_logic;
    lhs_ready: out std_logic;
    rhs_ready: out std_logic
  );
end entity;

-- Architecture of cmpf_single_precision
architecture arch of handshake_cmpf_0_inner is
  signal ip_lhs: std_logic_vector(32 + 1 downto 0);
  signal ip_rhs: std_logic_vector(32 + 1 downto 0);
begin
  join_inputs: entity work.handshake_cmpf_0_inner_join(arch)
    port map(
      -- inputs
      ins_valid(0)=> lhs_valid,
      ins_valid(1)=> rhs_valid,
      outs_ready=> result_ready,
      -- outputs
      outs_valid=> result_valid,
      ins_ready(0)=> lhs_ready,
      ins_ready(1)=> rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> lhs,
        --output
        R=> ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> rhs,
        --output
        R=> ip_rhs
    );
  operator: entity work.FPComparator_32bit(arch)
  port map (clk=> clk,
        ce=> '1',
        X=> ip_lhs,
        Y=> ip_rhs,
        unordered=> unordered,
        XltY=> XltY,
        XeqY=> XeqY,
        XgtY=> XgtY,
        XleY=> XleY,
        XgeY=> XgeY);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf
entity handshake_cmpf_0 is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(32 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(32 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

-- Architecture of cmpf
architecture arch of handshake_cmpf_0 is
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;
begin
  operator : entity work.handshake_cmpf_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      rhs => rhs,
      rhs_valid => rhs_valid,
      result_ready => result_ready,
      unordered => unordered,
      XltY => XltY,
      XeqY => XeqY,
      XgtY => XgtY,
      XleY => XleY,
      XgeY => XgeY,
      result_valid => result_valid,
      lhs_ready => lhs_ready,
      rhs_ready => rhs_ready
    );

  result(0) <= not unordered and XltY;
end architecture;

