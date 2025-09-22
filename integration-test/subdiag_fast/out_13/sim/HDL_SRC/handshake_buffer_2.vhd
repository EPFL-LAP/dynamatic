-- handshake_buffer_2 : buffer({'num_slots': 13, 'bitwidth': 0, 'buffer_type': 'FIFO_BREAK_NONE', 'extra_signals': {}, 'debug_counter': 0})



library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_dv_dataless
entity handshake_buffer_2_fifo is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of fifo_break_dv_dataless
architecture arch of handshake_buffer_2_fifo is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to 13 - 1;
  signal Head       : natural range 0 to 13 - 1;
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
          Tail <= (Tail + 1) mod 13;
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
          Head <= (Head + 1) mod 13;
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
          if ((Tail + 1) mod 13 = Head) then
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
          if ((Head + 1) mod 13 = Tail) then
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_none_dataless
entity handshake_buffer_2 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of fifo_break_none_dataless
architecture arch of handshake_buffer_2 is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
begin
  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;
  fifo_nready <= outs_ready;

  fifo : entity work.handshake_buffer_2_fifo(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  
end architecture;

