library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cmpf_single_precision is
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
    unordered    : out std_logic;
    XltY         : out std_logic;
    XeqY         : out std_logic;
    XgtY         : out std_logic;
    XleY         : out std_logic;
    XgeY         : out std_logic;
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of cmpf_single_precision is
  signal ip_lhs : std_logic_vector(32 + 1 downto 0);
  signal ip_rhs : std_logic_vector(32 + 1 downto 0);
begin
  join_inputs : entity work.join(arch) generic map(2)
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

  ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
    port map (
        --input
        X => lhs,
        --output
        R => ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
    port map (
        --input
        X => rhs,
        --output
        R => ip_rhs
    );
  operator : entity work.FPComparator_32bit(arch)
  port map (clk => clk,
        ce => '1',
        X => ip_lhs,
        Y => ip_rhs,
        unordered => unordered,
        XltY => XltY,
        XeqY => XeqY,
        XgtY => XgtY, 
        XleY => XleY, 
        XgeY => XgeY);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cmpf_double_precision is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(64 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(64 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    unordered    : out std_logic;
    XltY         : out std_logic;
    XeqY         : out std_logic;
    XgtY         : out std_logic;
    XleY         : out std_logic;
    XgeY         : out std_logic;
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end cmpf_double_precision;

architecture arch of cmpf_double_precision is
  signal join_valid : std_logic;
	signal buff_valid, oehb_valid, oehb_ready : std_logic;
	signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
  signal ip_lhs : std_logic_vector(64 + 1 downto 0);
  signal ip_rhs : std_logic_vector(64 + 1 downto 0);
begin

 oehb : entity work.oehb_dataless(arch)
  port map(
    clk        => clk,
    rst        => rst,
    ins_valid  => buff_valid,
    outs_ready => result_ready,
    outs_valid => result_valid,
    ins_ready  => oehb_ready
  );
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => buff_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_64bit(arch)
    port map (
        --input
        X => lhs,
        --output
        R => ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_64bit(arch)
    port map (
        --input
        X => rhs,
        --output
        R => ip_rhs
    );
  operator : entity work.FPComparator_64bit(arch)
  port map (clk => clk,
        ce => oehb_ready,
        X => ip_lhs,
        Y => ip_rhs,
        unordered => unordered,
        XltY => XltY,
        XeqY => XeqY,
        XgtY => XgtY, 
        XleY => XleY, 
        XgeY => XgeY);
end architecture;


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ENTITY_NAME is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of ENTITY_NAME is


  constant cmp_predicate : string := "COMPARATOR";
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;

begin

  gen_flopoco_ip : 
    if DATA_TYPE = 32 generate
      operator : entity work.cmpf_single_precision(arch)
      port map(
        clk, rst, lhs, lhs_valid,
        rhs, rhs_valid, result_ready,
        unordered, XltY, XeqY, XgtY, XleY, XgeY,
        result_valid, lhs_ready, rhs_ready);
    elsif DATA_TYPE = 64 generate
      operator : entity work.cmpf_double_precision(arch)
      port map(
        clk, rst, lhs, lhs_valid,
        rhs, rhs_valid, result_ready,
        unordered, XltY, XeqY, XgtY, XleY, XgeY,
        result_valid, lhs_ready, rhs_ready);
    else generate
      assert false
      report "cmpf must operate on 32bit or 64bit data!"
      severity failure;
    end generate;

  gen_result_signal :
    if cmp_predicate = "OEQ" generate
      result(0) <= not unordered and XeqY;
    elsif cmp_predicate = "OGT" generate
      result(0) <= not unordered and XgtY;
    elsif cmp_predicate = "OGE" generate
      result(0) <= not unordered and XgeY;
    elsif cmp_predicate = "OLT" generate
      result(0) <= not unordered and XltY;
    elsif cmp_predicate = "OLE" generate
      result(0) <= not unordered and XleY;
    elsif cmp_predicate = "ONE" generate
      result(0) <= not unordered and not XeqY;
    elsif cmp_predicate = "UEQ" generate
      result(0) <= unordered or XeqY;
    elsif cmp_predicate = "UGT" generate
      result(0) <= unordered or XgtY;
    elsif cmp_predicate = "UGE" generate
      result(0) <= unordered or XgeY;
    elsif cmp_predicate = "ULT" generate
      result(0) <= unordered or XltY;
    elsif cmp_predicate = "ULE" generate
      result(0) <= unordered or XleY;
    elsif cmp_predicate = "UNE" generate
      result(0) <= unordered or not XeqY;
    elsif cmp_predicate = "UNO" generate
      result(0) <= unordered;
    else generate 
      assert false
      report "COMPARATOR is an invalid predicate!"
      severity failure;
    end generate;

end architecture;
