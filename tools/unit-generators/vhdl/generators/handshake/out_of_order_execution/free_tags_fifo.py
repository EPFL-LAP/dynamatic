def generate_elastic_fifo_inner(name, params):
  size = params["size"]
  if "bitwidth" in params:
    bitwidth = params["bitwidth"]
  else:
    bitwidth = 0

  if bitwidth == 0:
    return _generate_elastic_fifo_inner_dataless(name, size)
  else:
    return _generate_elastic_fifo_inner(name, size, bitwidth)


def _generate_elastic_fifo_inner_dataless(name, size):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of elastic_fifo_inner_dataless
entity {name} is
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of elastic_fifo_inner_dataless
architecture arch of {name} is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to {size} - 1;
  signal Head       : natural range 0 to {size} - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;

begin

  -- ready if there is space in the fifo
  ins_ready <= not Full or outs_ready;

  -- read if next can accept and there is sth in fifo to read
  ReadEn <= (outs_ready and not Empty);

  outs_valid <= not Empty;

  WriteEn <= ins_valid and (not Full or outs_ready);

  -- valid
  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fifo_valid <= '0';
      else
        if (ReadEn = '1') then
          fifo_valid <= '1';
        elsif (outs_ready = '1') then
          fifo_valid <= '0';
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating tail
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Tail <= 0;
      else
        if (WriteEn = '1') then
          Tail <= (Tail + 1) mod {size};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Head <= 0;
      else
        if (ReadEn = '1') then
          Head <= (Head + 1) mod {size};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Full <= '0';
      else
        -- if only filling but not emptying
        if (WriteEn = '1') and (ReadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail + 1) mod {size} = Head) then
            Full <= '1';
          end if;
          -- if only emptying but not filling
        elsif (WriteEn = '0') and (ReadEn = '1') then
          Full <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then
          -- if new head index will reach tail index
          if ((Head + 1) mod {size} = Tail) then
            Empty <= '1';
          end if;
          -- if only filling but not emptying
        elsif (WriteEn = '1') and (ReadEn = '0') then
          Empty <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

  return entity + architecture


def _generate_elastic_fifo_inner(name, size, bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of elastic_fifo_inner
entity {name} is
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of elastic_fifo_inner
architecture arch of {name} is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to {size} - 1;
  signal Head       : natural range 0 to {size} - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;
  type FIFO_Memory is array (0 to {size} - 1) of std_logic_vector ({bitwidth} - 1 downto 0);
  signal Memory : FIFO_Memory;

begin

  -- ready if there is space in the fifo
  ins_ready <= not Full or outs_ready;
  -- read if next can accept and there is sth in fifo to read
  ReadEn <= (outs_ready and not Empty);
  outs_valid <= not Empty;
  outs <= Memory(Head);
  WriteEn <= ins_valid and (not Full or outs_ready);

  -- valid
  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fifo_valid <= '1';
      else
        if (ReadEn = '1') then
          fifo_valid <= '1';
        elsif (outs_ready = '1') then
          fifo_valid <= '0';
        end if;
      end if;
    end if;
  end process;

  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        for i in Memory'range loop
          Memory(i) <= std_logic_vector(to_unsigned(i, Memory(i)'length));
        end loop;
      else
        if (WriteEn = '1') then
          -- Write Data to Memory
          Memory(Tail) <= ins;
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating tail
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Tail <= {size} - 1;
      else
        if (WriteEn = '1') then
          Tail <= (Tail + 1) mod {size};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Head <= 0;
      else
        if (ReadEn = '1') then
          Head <= (Head + 1) mod {size};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Full <= '1';
      else
        -- if only filling but not emptying
        if (WriteEn = '1') and (ReadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail + 1) mod {size} = Head) then
            Full <= '1';
          end if;
          -- if only emptying but not filling
        elsif (WriteEn = '0') and (ReadEn = '1') then
          Full <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Empty <= '0';
      else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then
          -- if new head index will reach tail index
          if ((Head + 1) mod {size} = Tail) then
            Empty <= '1';
          end if;
          -- if only filling but not emptying
        elsif (WriteEn = '1') and (ReadEn = '0') then
          Empty <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

  return entity + architecture
  
def generate_free_tags_fifo(name, params):
  bitwidth = params["bitwidth"]
  fifo_depth = params["fifo_depth"]

  return _generate_free_tags_fifo(name, bitwidth, fifo_depth)


def _generate_free_tags_fifo(name, bitwidth, fifo_depth):
  fifo_name = f"{name}_fifo"

  dependencies = \
      generate_elastic_fifo_inner(fifo_name, {
          "size": fifo_depth,
          "bitwidth": bitwidth,
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of free_tags_fifo
entity {name} is 
port (
        clk           : in std_logic;
        rst           : in std_logic;
        ins           : in  std_logic_vector({bitwidth} - 1 downto 0);
        outs          : out std_logic_vector({bitwidth} - 1 downto 0);
        ins_valid     : in  std_logic;
        outs_ready    : in  std_logic;
        outs_valid    : out std_logic;
        ins_ready     : out std_logic
);
end entity;
"""
  architecture = f"""
-- Architecture of free_tags_fifo
architecture arch of {name} is
    signal mux_sel : std_logic;
    signal fifo_valid, fifo_ready : STD_LOGIC;
    signal fifo_pvalid, fifo_nready : STD_LOGIC;
    signal fifo_in, fifo_out: std_logic_vector({bitwidth}-1 downto 0);
begin
    
    process (mux_sel, fifo_out, ins) is
        begin
            if (mux_sel = '1') then
                outs <= fifo_out;
            else
                outs <= ins;
            end if;
    end process;

    outs_valid <= ins_valid or fifo_valid;    --fifo_valid is 0 only if fifo is empty
    ins_ready <= fifo_ready or outs_ready;
    fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid); --store in FIFO if next is not ins_ready or FIFO is already outputting something
    mux_sel <= fifo_valid;

    fifo_nready <= outs_ready;
    fifo_in <= ins;

    fifo: entity work.{fifo_name}(arch)
        port map (
        --inputs
            clk => clk, 
            rst => rst,
            ins =>fifo_in, 
            ins_valid  => fifo_pvalid, 
            outs_ready => fifo_nready,    
            
        --outputs
            outs => fifo_out,
            outs_valid => fifo_valid,
            ins_ready => fifo_ready   
        );
end architecture;
"""
  return dependencies + entity + architecture