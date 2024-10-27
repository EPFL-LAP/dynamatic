library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

entity mux is
  generic (
    SIZE         : integer;
    DATA_TYPE   : integer;
    SELECT_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  data_array(SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector(SELECT_TYPE - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of mux is
  signal tehb_ins                       : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
begin
  process (ins, ins_valid, outs_ready, index, index_valid, tehb_ins_ready)
    variable selectedData                   : std_logic_vector(DATA_TYPE - 1 downto 0);
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in SIZE - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready    <= (not index_valid) or (selectedData_valid and tehb_ins_ready);
    tehb_ins       <= selectedData;
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tehb_ins,
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

entity mux_with_tag is
  generic (
    SIZE         : integer;
    DATA_TYPE   : integer;
    SELECT_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  data_array(SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_spec_tag : in std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector(SELECT_TYPE - 1 downto 0);
    index_valid : in  std_logic;
    index_spec_tag : in std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- a bit complex and cannot override the original mux
architecture arch of mux_with_tag is
  signal index_inner : std_logic_vector(SELECT_TYPE - 1 downto 0);
  signal index_valid_inner : std_logic; 
  signal index_spec_tag_inner : std_logic;
  signal index_ready_inner : std_logic;
  signal tehb_ins                       : std_logic_vector(DATA_TYPE downto 0);
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
  signal outs_inner : std_logic_vector(DATA_TYPE downto 0);
begin

  index_buf : entity work.speculator_buffers(arch)
    generic map(
      DATA_TYPE => SELECT_TYPE,
      BUFFERS => 3
    )
    port map(
      clk => clk,
      rst => rst,
      ins       => index,
      ins_valid => index_valid,
      ins_spec_tag => index_spec_tag,
      ins_ready => index_ready,
      outs       => index_inner,
      outs_valid => index_valid_inner,
      outs_spec_tag => index_spec_tag_inner,
      outs_ready => index_ready_inner
    );

  outs <= outs_inner(DATA_TYPE - 1 downto 0);
  outs_spec_tag <= outs_inner(DATA_TYPE);
  process (ins, ins_valid, ins_spec_tag, outs_ready, index_inner, index_valid_inner, index_spec_tag_inner, tehb_ins_ready)
    variable selectedData                   : std_logic_vector(DATA_TYPE - 1 downto 0);
    variable selectedData_spec_tag          : std_logic;
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in SIZE - 1 downto 0 loop
      if unsigned(index_inner) = to_unsigned(i, index_inner'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid_inner and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_spec_tag := ins_spec_tag(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid_inner and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready_inner    <= (not index_valid_inner) or (selectedData_valid and tehb_ins_ready);
    tehb_ins       <= (selectedData_spec_tag or index_spec_tag_inner) & selectedData;
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => DATA_TYPE + 1 -- spec tag bit
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tehb_ins,
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs       => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
