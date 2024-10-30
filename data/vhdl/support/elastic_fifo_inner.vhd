library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity elastic_fifo_inner is
  generic (
    DATA_TYPE : integer;
    NUM_SLOTS     : integer
  );
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of elastic_fifo_inner is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to NUM_SLOTS - 1;
  signal Head       : natural range 0 to NUM_SLOTS - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;
  type FIFO_Memory is array (0 to NUM_SLOTS - 1) of std_logic_vector (DATA_TYPE - 1 downto 0);
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

  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        for i in Memory'range loop
          Memory(i) <= (others => '0');
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
        Tail <= 0;
      else
        if (WriteEn = '1') then
          Tail <= (Tail + 1) mod NUM_SLOTS;
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
          Head <= (Head + 1) mod NUM_SLOTS;
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
          if ((Tail + 1) mod NUM_SLOTS = Head) then
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
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then
          -- if new head index will reach tail index
          if ((Head + 1) mod NUM_SLOTS = Tail) then
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
