library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity end_sync is
  generic (
    BITWIDTH   : integer;
    MEM_INPUTS : integer
  );
  port (
    -- inputs
    clk, rst      : in std_logic;
    ins           : in data_array(0 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid     : in std_logic_vector(0 downto 0);
    memDone       : in data_array(MEM_INPUTS - 1 downto 0)(0 downto 0);
    memDone_valid : in std_logic_vector(MEM_INPUTS - 1 downto 0) := (others => '1');
    outs_ready    : in std_logic_vector(0 downto 0);
    -- outputs
    outs          : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
    outs_valid    : out std_logic_vector(0 downto 0);
    ins_ready     : out std_logic_vector(0 downto 0);
    memDone_ready : out std_logic_vector(MEM_INPUTS - 1 downto 0)
  );
end entity;

architecture arch of end_sync is
  signal allPValid : std_logic;
  signal nReady    : std_logic;
  signal valid     : std_logic;
  signal mem_valid : std_logic;
  signal joinValid : std_logic;
  signal joinReady : std_logic;
  signal out_array : std_logic_vector(1 downto 0);

begin
  nReady    <= out_array(0);
  joinReady <= out_array(1);

  process (ins_valid, ins)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned((ins(0)));
    tmp_valid_out := '0';
    if (ins_valid(0) = '1') then
      tmp_data_out  := unsigned(ins(0));
      tmp_valid_out := ins_valid(0);
    end if;
    outs(0) <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid   <= tmp_valid_out;
  end process;

  mem_and : entity work.and_n(vanilla) generic map (MEM_INPUTS)
    port map(memDone_valid, mem_valid);

  j : entity work.join(arch) generic map(2)
    port map(
    (valid, mem_valid),
      outs_ready(0),
      joinValid,
      out_array);

  outs_valid(0) <= joinValid;
  ins_ready(0)  <= joinReady;
end architecture;
