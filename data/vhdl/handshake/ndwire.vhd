library ieee;
use ieee.std_logic_1164.all;

entity ndwire is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of br is
  type nd_state_t is (SLEEPING, RUNNING);
  signal state, next_state : nd_state_t;

  -- This is the source of non-determism.
  -- E.g. it needs to be set to a primary input in a formal tool
  signal nd_next_state : nd_state_t;

begin

  process (clk, rst)
  begin
    -- The initialization of the state is non-deterministic
    if rst = '1' then
      state <= nd_next_state;
    elsif rising_edge(clk) then
      state <= next_state;
    end if;
  end process;

  process (state, ins_valid, outs_ready)
  begin
    -- If the wire is sleeping it can always switch to the running state.
    -- If (ins_valid and outs_ready) we either have a transaction
    -- and can freely choose the state again.
    if (state = SLEEPING) then
      next_state <= nd_next_state;
    elsif (ins_valid and outs_ready) then
      next_state <= nd_next_state;
    else
      next_state <= state;
    end if;
  end process;

  ins_ready <= outs_ready and (state = RUNNING);
  outs_valid <= ins_valid and (state = RUNNING);
  outs <= ins;

end architecture;