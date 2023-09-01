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
