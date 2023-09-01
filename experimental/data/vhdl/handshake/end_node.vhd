library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity end_node is
  generic (
    MEM_INPUTS : integer;
    BITWIDTH   : integer
  );

  port (
    -- inputs
    ins             : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid       : in std_logic;
    mems_done_valid : in std_logic_vector(MEM_INPUTS - 1 downto 0) := (others => '1');
    clk             : in std_logic;
    rst             : in std_logic;
    outs_ready      : in std_logic;
    -- outputs
    ins_ready       : out std_logic;
    mems_done_ready : out std_logic_vector(MEM_INPUTS - 1 downto 0);
    outs            : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid      : out std_logic);
end end_node;

architecture arch of end_node is
  signal allPValid : std_logic;
  signal nReady    : std_logic;
  signal valid     : std_logic;
  signal mem_valid : std_logic;
  signal joinValid : std_logic;
  signal joinReady : std_logic;

begin
  process (pValid, ins)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;

  begin
    tmp_data_out  := unsigned(ins);
    tmp_valid_out := '0';

    if (pValid = '1') then
      tmp_data_out  := unsigned(ins);
      tmp_valid_out := ins_valid;
    end if;

    outs  <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid <= tmp_valid_out;
  end process;

  mem_and : entity work.andN(vanilla) generic map (MEM_INPUTS)
    port map(mems_done_valid, mem_valid);

  j : entity work.join(arch) generic map(2)
    port map(
    (valid, mem_valid),
      outs_ready,
      joinValid,
      (nReady, joinReady));
  outs_valid <= joinValid;

  process (joinReady)
  begin
    ins_ready <= joinReady;
  end process;
end architecture;
