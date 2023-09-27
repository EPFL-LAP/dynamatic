library ieee;
use ieee.std_logic_1164.all;
package customTypes is

  type data_array is array(natural range <>) of std_logic_vector;

end package;

library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity andN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(n - 1 downto 0);
    res : out std_logic);
end andN;

architecture vanilla of andn is
  signal dummy : std_logic_vector(n - 1 downto 0);
begin
  dummy <= (others => '1');
  res   <= '1' when x = dummy else
    '0';
end vanilla;

library ieee;
use ieee.std_logic_1164.all;

entity join is generic (SIZE : integer);
port (
  ins_valid  : in std_logic_vector(SIZE - 1 downto 0);
  outs_ready : in std_logic;
  outs_valid : out std_logic;
  ins_ready  : out std_logic_vector(SIZE - 1 downto 0));
end join;

architecture arch of join is
  signal allPValid : std_logic;

begin

  allPValidAndGate : entity work.andN generic map(SIZE)
    port map(
      ins_valid,
      allPValid);

  outs_valid <= allPValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(SIZE - 1 downto 0);
  begin
    for i in 0 to SIZE - 1 loop
      singlePValid(i) := '1';
      for j in 0 to SIZE - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to SIZE - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity TEHB is
  generic (
    BITWIDTH : integer
  );
  port (
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    outs_valid : out std_logic;
    ins_ready  : out std_logic);
end TEHB;

architecture arch of TEHB is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      full_reg <= '0';

    elsif (rising_edge(clk)) then
      full_reg <= outs_valid and not outs_ready;

    end if;
  end process;

  process (clk, rst) is

  begin
    if (rst = '1') then
      data_reg <= (others => '0');

    elsif (rising_edge(clk)) then
      if (reg_en) then
        data_reg <= ins;
      end if;

    end if;
  end process;

  process (mux_sel, data_reg, ins) is
  begin
    if (mux_sel = '1') then
      outs <= data_reg;
    else
      outs <= ins;
    end if;
  end process;
  outs_valid <= ins_valid or full_reg;
  ins_ready  <= not full_reg;
  reg_en     <= ins_ready and ins_valid and not outs_ready;
  mux_sel    <= full_reg;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity OEHB is
  generic (
    BITWIDTH : integer
  );
  port (
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    outs_valid : out std_logic;
    ins_ready  : out std_logic);
end OEHB;

architecture arch of OEHB is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      outs_valid <= '0';

    elsif (rising_edge(clk)) then
      outs_valid <= ins_valid or not ins_ready;

    end if;
  end process;

  process (clk, rst) is

  begin
    if (rst = '1') then
      data_reg <= (others => '0');

    elsif (rising_edge(clk)) then
      if (reg_en) then
        data_reg <= ins;
      end if;

    end if;
  end process;
  ins_ready <= not outs_valid or outs_ready;
  reg_en    <= ins_ready and ins_valid;
  outs      <= data_reg;

end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity antitokens is port (
  clk, reset                 : in std_logic;
  pvalid1, pvalid0           : in std_logic;
  kill1, kill0               : out std_logic;
  generate_at1, generate_at0 : in std_logic;
  stop_valid                 : out std_logic);

end antitokens;

architecture arch of antitokens is

  signal reg_in0, reg_in1, reg_out0, reg_out1 : std_logic;

begin

  reg0 : process (clk, reset, reg_in0)
  begin
    if (reset = '1') then
      reg_out0 <= '0';
    else
      if (rising_edge(clk)) then
        reg_out0 <= reg_in0;
      end if;
    end if;
  end process reg0;

  reg1 : process (clk, reset, reg_in1)
  begin
    if (reset = '1') then
      reg_out1 <= '0';
    else
      if (rising_edge(clk)) then
        reg_out1 <= reg_in1;
      end if;
    end if;
  end process reg1;

  reg_in0 <= not pvalid0 and (generate_at0 or reg_out0);
  reg_in1 <= not pvalid1 and (generate_at1 or reg_out1);

  stop_valid <= reg_out0 or reg_out1;

  kill0 <= generate_at0 or reg_out0;
  kill1 <= generate_at1 or reg_out1;
end arch;
library ieee;
use ieee.std_logic_1164.all;

entity branchSimple is port (
  condition,
  pValid      : in std_logic;
  nReadyArray : in std_logic_vector(1 downto 0);
  validArray  : out std_logic_vector(1 downto 0);
  ready       : out std_logic);
end branchSimple;

architecture arch of branchSimple is
begin
  validArray(1) <= (not condition) and pValid;
  validArray(0) <= condition and pValid;

  ready <= (nReadyArray(1) and not condition)
    or (nReadyArray(0) and condition);

end arch;

library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

entity delay_buffer is
  generic (
    SIZE : integer := 32
  );
  port (
    clk, rst  : in std_logic;
    valid_in  : in std_logic;
    ready_in  : in std_logic;
    valid_out : out std_logic);
end entity;

architecture arch of delay_buffer is

  type mem is array (SIZE - 1 downto 0) of std_logic;
  signal regs : mem;

begin

  gen_assignements : for i in 0 to SIZE - 1 generate
    first_assignment : if i = 0 generate
      process (clk) begin
        if rising_edge(clk) then
          if (ready_in = '1' or rst = '1') then
            regs(i) <= valid_in;
          end if;
        end if;
      end process;
    end generate first_assignment;

    other_assignments : if i > 0 generate
      process (clk) begin
        if rising_edge(clk) then
          if (rst = '1') then
            regs(i) <= '0';
          elsif (ready_in = '1') then
            regs(i) <= regs(i - 1);
          end if;
        end if;
      end process;
    end generate other_assignments;

  end generate gen_assignements;

  valid_out <= regs(SIZE - 1);
end architecture;
library ieee;
use ieee.std_logic_1164.all;

entity eagerFork_RegisterBLock is
  port (
    clk, reset,
    p_valid, n_stop,
    p_valid_and_fork_stop : in std_logic;
    valid, block_stop     : out std_logic);
end eagerFork_RegisterBLock;

architecture arch of eagerFork_RegisterBLock is
  signal reg_value, reg_in, block_stop_internal : std_logic;
begin

  block_stop_internal <= n_stop and reg_value;

  block_stop <= block_stop_internal;

  reg_in <= block_stop_internal or (not p_valid_and_fork_stop);

  valid <= reg_value and p_valid;

  reg : process (clk, reset, reg_in)
  begin
    if (reset = '1') then
      reg_value <= '1'; --contains a "stop" signal - must be 1 at reset
    else
      if (rising_edge(clk)) then
        reg_value <= reg_in;
      end if;
    end if;
  end process reg;
end arch;
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity elasticBufferDummy is
  generic (
    SIZE          : integer;
    INPUTS        : integer := 32;
    DATA_SIZE_IN  : integer;
    DATA_SIZE_OUT : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    dataOutArray : out std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
    ReadyArray   : out std_logic_vector(0 downto 0);
    ValidArray   : out std_logic_vector(0 downto 0);
    nReadyArray  : in std_logic_vector(0 downto 0);
    pValidArray  : in std_logic_vector(0 downto 0));
end elasticBufferDummy;

architecture arch of elasticBufferDummy is

begin

  dataOutArray  <= dataInArray;
  ValidArray(0) <= pValidArray(0);
  ReadyArray(0) <= nReadyArray(0);

end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity buffer_fifo is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end buffer_fifo;

architecture arch of buffer_fifo is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      full_reg <= '0';

    elsif (rising_edge(clk)) then
      full_reg <= outs_valid and not outs_ready;

    end if;
  end process;

  process (clk, rst) is

  begin
    if (rst = '1') then
      data_reg <= (others => '0');

    elsif (rising_edge(clk)) then
      if (reg_en) then
        data_reg <= ins;
      end if;

    end if;
  end process;

  process (mux_sel, data_reg, ins) is
  begin
    if (mux_sel = '1') then
      outs <= data_reg;
    else
      outs <= ins;
    end if;
  end process;
  outs_valid <= ins_valid or full_reg;
  ins_ready  <= not full_reg;
  reg_en     <= ins_ready and ins_valid and not outs_ready;
  mux_sel    <= full_reg;
end arch;
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is

  generic (
    INPUTS   : integer;
    BITWIDTH : integer);
  port (
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector(INPUTS - 1 downto 0);
    outs_ready : in std_logic;
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(INPUTS - 1 downto 0));
end merge_notehb;

architecture arch of merge_notehb is
  signal tehb_data_in : std_logic_vector(BITWIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin

  process (ins_valid, ins)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    for I in INPUTS - 1 downto 0 loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := ins_valid(I);
      end if;
    end loop;

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    tehb_pvalid  <= tmp_valid_out;

  end process;

  process (tehb_ready)
  begin
    for I in 0 to INPUTS - 1 loop
      ins_ready(I) <= tehb_ready;
    end loop;
  end process;

  tehb_ready <= outs_ready;
  outs_valid <= tehb_pvalid;
  outs       <= tehb_data_in;

end arch;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity mul_4_stage is
  generic (
    BITWIDTH : integer
  );
  port (
    clk : in std_logic;
    ce  : in std_logic;
    a   : in std_logic_vector(BITWIDTH - 1 downto 0);
    b   : in std_logic_vector(BITWIDTH - 1 downto 0);
    p   : out std_logic_vector(BITWIDTH - 1 downto 0));
end entity;

architecture behav of mul_4_stage is

  signal a_reg : std_logic_vector(BITWIDTH - 1 downto 0);
  signal b_reg : std_logic_vector(BITWIDTH - 1 downto 0);
  signal q0    : std_logic_vector(BITWIDTH - 1 downto 0);
  signal q1    : std_logic_vector(BITWIDTH - 1 downto 0);
  signal q2    : std_logic_vector(BITWIDTH - 1 downto 0);
  signal mul   : std_logic_vector(BITWIDTH - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), BITWIDTH));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;

end architecture;
library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity nandN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end nandN;

architecture arch of nandn is
  signal dummy  : std_logic_vector(n - 1 downto 0);
  signal andRes : std_logic;
begin
  dummy  <= (others => '1');
  andRes <= '1' when x = dummy else
    '0';
  res <= not andRes;
end arch;
library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity norN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end norN;

architecture arch of norN is
  signal dummy : std_logic_vector(n - 1 downto 0);
  signal orRes : std_logic;
begin
  dummy <= (others => '0');
  orRes <= '0' when x = dummy else
    '1';
  res <= not orRes;
end arch;

library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity orN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end orN;

architecture vanilla of orN is
  signal dummy : std_logic_vector(n - 1 downto 0);
begin
  dummy <= (others => '0');
  res   <= '0' when x = dummy else
    '1';
end vanilla;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_priority is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests (pValid signals)
        data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready from next
        priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0) -- priority function output
    );
end entity;

architecture arch of read_priority is

begin
    process(req, data_ready)
        variable prio_req : std_logic;
    begin
        -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
        priority_out(0) <= req(0) and data_ready(0);
        prio_req        := req(0) and data_ready(0);
        for I in 1 to ARBITER_SIZE - 1 loop
            priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
            prio_req        := prio_req or (req(I) and data_ready(I));
        end loop;
    end process;
end architecture;

--------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_address_mux is
    generic(
        ARBITER_SIZE : natural;
        ADDR_WIDTH   : natural
    );
    port(
        sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        addr_out : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
    );
end entity;

architecture arch of read_address_mux is

begin
    process(sel, addr_in)
        variable addr_out_var : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    begin
        addr_out_var := (others => '0');
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                addr_out_var := addr_in(I);
            end if;
        end loop;
        addr_out     <= addr_out_var;
    end process;
end architecture;

--------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_address_ready is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of read_address_ready is
begin
    GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
        ready(I) <= nReady(I) and sel(I);
    end generate GEN1;
end architecture;

--------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_data_signals is
    generic(
        ARBITER_SIZE : natural;
        DATA_WIDTH   : natural
    );
    port(
        rst       : in  std_logic;
        clk       : in  std_logic;
        sel       : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        read_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        out_data  : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
        valid     : out std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of read_data_signals is
    signal sel_prev : std_logic_vector(ARBITER_SIZE - 1 downto 0);
    signal out_reg: data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
begin

    process(clk, rst) is
    begin
        if (rst = '1') then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I)    <= '0';
                sel_prev(I) <= '0';
            end loop;
        elsif (rising_edge(clk)) then
            for I in 0 to ARBITER_SIZE - 1 loop
                sel_prev(I) <= sel(I);
                if (sel(I) = '1') then
                    valid(I)    <= '1';  --or not nReady(I); -- just sel(I) ??
                    --sel_prev(I) <= '1';
                else
                    if (nReady(I) = '1') then
                        valid(I)  <= '0';
                        ---sel_prev(I) <= '0';
                    end if;
                end if;
            end loop;
        end if;
    end process;

    process(clk, rst) is
    begin
        if (rising_edge(clk)) then
         for I in 0 to ARBITER_SIZE - 1 loop
                if (sel_prev(I) = '1') then
                    out_reg(I) <= read_data;
                end if;
         end loop;
         end if;
    end process;

    process(read_data, sel_prev, out_reg) is
    begin
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel_prev(I) = '1') then
                out_data(I) <= read_data;
            else
                out_data(I) <= out_reg(I);
            end if;
        end loop;
    end process;

end architecture;



--------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_memory_arbiter is
    generic(
        ARBITER_SIZE : natural := 2;
        ADDR_WIDTH   : natural := 32;
        DATA_WIDTH   : natural := 32
    );
    port(
        rst              : in  std_logic;
        clk              : in  std_logic;
        --- interface to previous
        pValid           : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests
        ready            : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready to process read
        address_in       : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        ---interface to next
        nReady           : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can accept data
        valid            : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- sending data to next component
        data_out         : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data to next components

        ---interface to memory
        read_enable      : out std_logic;
        read_address     : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
        data_from_memory : in  std_logic_vector(DATA_WIDTH - 1 downto 0));

end entity;

architecture arch of read_memory_arbiter is
    signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

    priority : entity work.read_priority
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            req          => pValid,
            data_ready   => nReady,
            priority_out => priorityOut
        );

    addressing : entity work.read_address_mux
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            ADDR_WIDTH   => ADDR_WIDTH
        )
        port map(
            sel      => priorityOut,
            addr_in  => address_in,
            addr_out => read_address
        );

    adderssReady : entity work.read_address_ready
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            sel    => priorityOut,
            nReady => nReady,
            ready  => ready
        );

    data : entity work.read_data_signals
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            DATA_WIDTH   => DATA_WIDTH
        )
        port map(
            rst       => rst,
            clk       => clk,
            sel       => priorityOut,
            read_data => data_from_memory,
            out_data  => data_out,
            valid     => valid,
            nReady    => nReady
        );

    process(priorityOut) is
        variable read_en_var : std_logic;
    begin
        read_en_var := '0';
        for I in 0 to ARBITER_SIZE - 1 loop
            read_en_var := read_en_var or priorityOut(I);
        end loop;
        read_enable <= read_en_var;
    end process;

end architecture;


--------------------------------------------------

--------------------------------------------------
---write_priority----

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_priority is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of write_priority is

begin

    process(data_ready, req)
        variable prio_req : std_logic;

    begin
        -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
        priority_out(0) <= req(0) and data_ready(0);
        prio_req        := req(0) and data_ready(0);

        for I in 1 to ARBITER_SIZE - 1 loop
            priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
            prio_req        := prio_req or (req(I) and data_ready(I));
        end loop;
    end process;
end architecture;

--------------------------------------------------
---write_address_mux----

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_address_mux is
    generic(
        ARBITER_SIZE : natural;
        ADDR_WIDTH   : natural
    );
    port(
        sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        addr_out : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
    );
end entity;

architecture arch of write_address_mux is

begin
    process(sel, addr_in)
        variable addr_out_var : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    begin
        addr_out_var := (others => '0');
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                addr_out_var := addr_in(I);
            end if;
        end loop;
        addr_out     <= addr_out_var;
    end process;
end architecture;

--------------------------------------------------
---write_address_ready----

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_address_ready is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );

end entity;

architecture arch of write_address_ready is

begin

    GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
        ready(I) <= nReady(I) and sel(I);
    end generate GEN1;

end architecture;

--------------------------------------------------
---write_data_signals----

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_data_signals is
    generic(
        ARBITER_SIZE : natural;
        DATA_WIDTH   : natural
    );
    port(
        rst        : in  std_logic;
        clk        : in  std_logic;
        sel        : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        write_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        in_data    : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
        valid      : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );

end entity;

architecture arch of write_data_signals is

begin

    process(sel, in_data)
        variable data_out_var : std_logic_vector(DATA_WIDTH - 1 downto 0);
    begin
        data_out_var := (others => '0');

        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                data_out_var := in_data(I);
            end if;
        end loop;
        write_data <= data_out_var;
    end process;

    process(clk, rst) is
    begin
        if (rst = '1') then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I) <= '0';
            end loop;

        elsif (rising_edge(clk)) then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I) <= sel(I);
            end loop;
        end if;
    end process;
end architecture;

--------------------------------------------------
---write_memory_arbiter----

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_memory_arbiter is
    generic(
        ARBITER_SIZE : natural := 2;
        ADDR_WIDTH   : natural := 32;
        DATA_WIDTH   : natural := 32
    );
    port(
        rst            : in  std_logic;
        clk            : in  std_logic;
        --- interface to previous
        pValid         : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); --write requests
        ready          : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready
        address_in     : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        data_in        : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data from previous that want to write

        ---interface to next
        nReady         : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can continue after write
        valid          : out std_logic_vector(ARBITER_SIZE - 1 downto 0); --sending write confirmation to next component

        ---interface to memory
        write_enable   : out std_logic;
        enable         : out std_logic;
        write_address  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
        data_to_memory : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );

end entity;

architecture arch of write_memory_arbiter is
    signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

    priority : entity work.write_priority
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            req          => pValid,
            data_ready   => nReady,
            priority_out => priorityOut
        );

    addressing : entity work.write_address_mux
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            ADDR_WIDTH   => ADDR_WIDTH
        )
        port map(
            sel      => priorityOut,
            addr_in  => address_in,
            addr_out => write_address
        );

    addressReady : entity work.write_address_ready
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            sel    => priorityOut,
            nReady => nReady,
            ready  => ready
        );
    data : entity work.write_data_signals
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            DATA_WIDTH   => DATA_WIDTH
        )
        port map(
            rst        => rst,
            clk        => clk,
            sel        => priorityOut,
            write_data => data_to_memory,
            in_data    => data_in,
            valid      => valid
        );

    process(priorityOut) is
        variable write_en_var : std_logic;
    begin
        write_en_var := '0';
        for I in 0 to ARBITER_SIZE - 1 loop
            write_en_var := write_en_var or priorityOut(I);
        end loop;
        write_enable <= write_en_var;
        enable       <= write_en_var;
    end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity buffer_seq is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end buffer_seq;

architecture arch of buffer_seq is

  signal tehb1_valid, tehb1_ready     : std_logic;
  signal oehb1_valid, oehb1_ready     : std_logic;
  signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  tehb1 : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => oehb1_ready,
      outs_valid => tehb1_valid,

      ins_ready => tehb1_ready,
      ins       => ins,
      outs      => tehb1_dataOut
    );

  oehb1 : entity work.OEHB(arch) generic map (BITWIDTH)
    port map(

      clk        => clk,
      rst        => rst,
      ins_valid  => tehb1_valid,
      outs_ready => outs_ready,
      outs_valid => oehb1_valid,

      ins_ready => oehb1_ready,
      ins       => tehb1_dataOut,
      outs      => oehb1_dataOut
    );

  outs       <= oehb1_dataOut;
  outs_valid <= oehb1_valid;
  ins_ready  <= tehb1_ready;

end arch;