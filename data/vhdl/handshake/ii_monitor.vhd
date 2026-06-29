library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ii_monitor is
  generic (
    -- Bit-width of the dominating control_merge's index output data signal.
    INDEX_WIDTH     : integer;
    -- Index value identifying a back-edge (loop-internal) activation.
    LOOP_BACK_INDEX : integer
  );
  port (
    clk         : in std_logic;
    rst         : in std_logic;
    -- Index channel of the dominating control_merge (read-only).
    index       : in std_logic_vector(INDEX_WIDTH - 1 downto 0);
    index_valid : in std_logic;
    index_ready : in std_logic;
    -- An exit channel of the loop (read-only).
    exit_valid  : in std_logic;
    exit_ready  : in std_logic
  );
end entity;

architecture arch of ii_monitor is
begin
  process (clk)
    variable cycle     : integer := 0;
    variable measuring : boolean := false;
    variable start_cyc : integer := 0;
    variable last_cyc  : integer := 0;
    variable iters     : integer := 0;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        cycle     := 0;
        measuring := false;
        start_cyc := 0;
        last_cyc  := 0;
        iters     := 0;
      else
        cycle := cycle + 1;

        -- Observe the control_merge index channel.
        if index_valid = '1' and index_ready = '1' then
          if to_integer(unsigned(index)) /= LOOP_BACK_INDEX then
            -- Activation from outside the loop (first entry or re-entry).
            if measuring and iters > 1 then
              report "II_INSTRUMENT: II=" &
                     real'image(real(last_cyc - start_cyc) / real(iters - 1)) &
                     " iterations=" & integer'image(iters) severity note;
            end if;
            measuring := true;
            start_cyc := cycle;
            last_cyc  := cycle;
            iters     := 1;
          else
            -- Activation from the loop back-edge.
            if measuring then
              last_cyc := cycle;
              iters    := iters + 1;
            end if;
          end if;
        end if;

        -- Observe the loop exit channel.
        if exit_valid = '1' and exit_ready = '1' then
          if measuring and iters > 1 then
            report "II_INSTRUMENT: II=" &
                   real'image(real(last_cyc - start_cyc) / real(iters - 1)) &
                   " iterations=" & integer'image(iters) severity note;
          end if;
          measuring := false;
        end if;
      end if;
    end if;
  end process;
end architecture;
