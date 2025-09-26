-- handshake_mux_0 : mux({'size': 2, 'data_bitwidth': 8, 'index_bitwidth': 1, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

-- Entity of mux
entity handshake_mux_0_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- data input channels
    ins       : in  data_array(2 - 1 downto 0)(9 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector(1 - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(9 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of mux
architecture arch of handshake_mux_0_inner is
begin
  process (ins, ins_valid, outs_ready, index, index_valid)
    variable selectedData                   : std_logic_vector(9 - 1 downto 0);
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in 2 - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and outs_ready) or (not ins_valid(i));
    end loop;

    index_ready <= (not index_valid) or (selectedData_valid and outs_ready);
    outs        <= selectedData;
    outs_valid  <= selectedData_valid;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_mux_0 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in data_array(2 - 1 downto 0)(8 - 1 downto 0);
    ins_valid : in std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    ins_0_spec : in std_logic_vector(1 - 1 downto 0);
    ins_1_spec : in std_logic_vector(1 - 1 downto 0);
    index : in std_logic_vector(1 - 1 downto 0);
    index_valid : in std_logic;
    index_ready : out std_logic;
    index_spec : in std_logic_vector(1 - 1 downto 0);
    outs : out std_logic_vector(8 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (mux)
architecture arch of handshake_mux_0 is
  signal ins_inner : data_array(1 downto 0)(8 downto 0);
  signal ins_inner_valid : std_logic_vector(1 downto 0);
  signal ins_inner_ready : std_logic_vector(1 downto 0);
  signal outs_inner_concat : std_logic_vector(8 downto 0);
  signal outs_inner_concat_valid : std_logic;
  signal outs_inner_concat_ready : std_logic;
  signal outs_inner : std_logic_vector(7 downto 0);
  signal outs_inner_valid : std_logic;
  signal outs_inner_ready : std_logic;
  signal outs_inner_spec : std_logic_vector(0 downto 0);
begin
  -- Concatenate data and extra signals
  ins_inner(0)(8 - 1 downto 0) <= ins(0);
  ins_inner(0)(8 downto 8) <= ins_0_spec;
  ins_inner(1)(8 - 1 downto 0) <= ins(1);
  ins_inner(1)(8 downto 8) <= ins_1_spec;
  ins_inner_valid <= ins_valid;
  ins_ready <= ins_inner_ready;
  outs_inner <= outs_inner_concat(8 - 1 downto 0);
  outs_inner_spec <= outs_inner_concat(8 downto 8);
  outs_inner_valid <= outs_inner_concat_valid;
  outs_inner_concat_ready <= outs_inner_ready;

  -- Forwarding logic
  outs_spec <= index_spec or outs_inner_spec;

  outs <= outs_inner;
  outs_valid <= outs_inner_valid;
  outs_inner_ready <= outs_ready;

  inner : entity work.handshake_mux_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_inner_valid,
      ins_ready => ins_inner_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready,
      outs => outs_inner_concat,
      outs_valid => outs_inner_concat_valid,
      outs_ready => outs_inner_concat_ready
    );
end architecture;

