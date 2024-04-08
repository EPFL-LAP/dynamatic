library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity end is
  generic (
    BITWIDTH   : integer;
    MEM_INPUTS : integer
  );
  port (
    -- inputs
    clk, rst      : in std_logic;
    ins           : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid     : in std_logic;
    memDone_valid : in std_logic_vector(MEM_INPUTS - 1 downto 0) := (others => '1');
    outs_ready    : in std_logic;
    -- outputs
    outs          : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid    : out std_logic;
    ins_ready     : out std_logic;
    memDone_ready : out std_logic_vector(MEM_INPUTS - 1 downto 0)
  );
end entity;

architecture arch of end is
  signal allPValid : std_logic;
  signal nReady : std_logic;
  signal valid : std_logic;
  signal mem_valid : std_logic;
  signal joinValid : std_logic;
  signal joinReady : std_logic;
  signal out_array : std_logic_vector(1 downto 0);

begin
  nReady <= out_array(0);
  joinReady <= out_array(1);

  process (ins_valid, ins)
    variable tmp_data_out : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out := unsigned(ins);
    tmp_valid_out := '0';
    if (ins_valid = '1') then
      tmp_data_out := unsigned(ins);
      tmp_valid_out := ins_valid;
    end if;
    outs <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid <= tmp_valid_out;
  end process;

  mem_and : entity work.andN(vanilla) generic map (MEM_INPUTS)
    port map(memDone_valid, mem_valid);

  j : entity work.join(arch) generic map(2)
    port map(
    (valid, mem_valid),
      outs_ready,
      joinValid,
      out_array);
  outs_valid <= joinValid;

  process (joinReady)
  begin
    ins_ready <= joinReady;
  end process;
end architecture;
