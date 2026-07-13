from generators.support.utils import ExtraSignals


def generate_ii_monitor(name, params):
    # Nesting depth of the measured loop (1 for a top-level loop) and deepest
    # depth reachable from it through its own descendants (equal to
    # loop_depth when the loop is innermost).
    loop_depth = params["loop_depth"]
    loop_max_depth = params["loop_max_depth"]
    # Extra signals carried by the observed channels (e.g. a "spec" signal under
    # speculation). The monitor ignores their values but must declare matching
    # ports so that its instantiation in the parent module is well-formed.
    entry_extra_signals = params.get("entry_extra_signals", None) or {}
    backedge_extra_signals = params.get("backedge_extra_signals", None) or {}
    exit_extra_signals = params.get("exit_extra_signals", None) or {}

    return _generate_ii_monitor(name, loop_depth, loop_max_depth,
                                entry_extra_signals, backedge_extra_signals,
                                exit_extra_signals)


def _extra_signal_ports(channel_name: str,
                        extra_signals: ExtraSignals) -> list[str]:
    return [
        f"{channel_name}_{signal_name} : in std_logic_vector({signal_width} - 1 downto 0)"
        for signal_name, signal_width in extra_signals.items()
    ]


def _generate_ii_monitor(name: str, loop_depth: int, loop_max_depth: int,
                         entry_extra_signals: ExtraSignals,
                         backedge_extra_signals: ExtraSignals,
                         exit_extra_signals: ExtraSignals) -> str:
    # Assemble the port list. All ports are inputs so the monitor never drives
    # the circuit (it only taps existing wires). The three observed channels are
    # control channels (no data), so only their valid/ready wires are declared.
    ports = [
        "clk            : in std_logic",
        "rst            : in std_logic",
        # Entry channel: a control merge input fed from outside the loop.
        "entry_valid    : in std_logic",
        "entry_ready    : in std_logic",
        *_extra_signal_ports("entry", entry_extra_signals),
        # Back-edge channel: a control merge input fed from inside the loop.
        "backedge_valid : in std_logic",
        "backedge_ready : in std_logic",
        *_extra_signal_ports("backedge", backedge_extra_signals),
        # Exit channel: a loop-exit branch result leaving the loop.
        "exit_valid     : in std_logic",
        "exit_ready     : in std_logic",
        *_extra_signal_ports("exit", exit_extra_signals),
    ]
    port_decls = ";\n    ".join(ports)

    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- II monitor: simulation-only component that measures the initiation interval
-- (II) of a loop at nesting depth {loop_depth} (of {loop_max_depth} in its
-- nest) by observing the loop header control merge's input channels and the
-- loop-exit channel.
--
-- All ports are inputs so that the monitor passively reads existing signals
-- without driving any of them, avoiding multiple-driver conflicts with the
-- actual circuit. Any extra signals on the observed channels are declared (so
-- the instantiation matches) but left unused.
--
-- Measurement logic:
--   * entry channel fires (entry_valid and entry_ready): activation from
--     outside the loop; report the previous window (if one was open) and start
--     a fresh measurement window.
--   * back-edge channel fires (backedge_valid and backedge_ready): advance the
--     window by one iteration.
--   * exit channel fires (exit_valid and exit_ready): report the current window
--     (if one was open) and stop measuring.
--
-- A window's iteration count is always reported (including for loops that ran
-- 0 or 1 times); the II is only reported when more than one iteration was seen
-- (otherwise there is no interval to measure and "n/a" is printed instead).
entity {name} is
  port (
    {port_decls}
  );
end entity;

architecture arch of {name} is
begin
  process (clk)
    variable cycle     : integer := 0;
    variable measuring : boolean := false;
    variable start_cyc : integer := 0;
    variable last_cyc  : integer := 0;
    variable iters     : integer := 0;

    -- Report a finished measurement window. The iteration count is always
    -- printed; the II is only printed when more than one iteration was seen.
    procedure report_window(iters     : integer;
                            start_cyc : integer;
                            last_cyc  : integer) is
    begin
      if iters > 1 then
        report "II_INSTRUMENT: loop=" & {name}'path_name &
               " depth={loop_depth}/{loop_max_depth} II=" &
               real'image(real(last_cyc - start_cyc) / real(iters - 1)) &
               " iterations=" & integer'image(iters) severity note;
      else
        report "II_INSTRUMENT: loop=" & {name}'path_name &
               " depth={loop_depth}/{loop_max_depth} II=n/a iterations=" &
               integer'image(iters) severity note;
      end if;
    end procedure;
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

        -- The three channels are processed in the order back-edge, exit, entry
        -- so that same-cycle collisions resolve correctly. Within one activation
        -- the last back-edge may coincide with the exit (count it, then close);
        -- at a seam the exit of one activation may coincide with the entry of
        -- the next (close the old window before the entry opens a new one). The
        -- entry and back-edge are the two inputs of the same control merge, so
        -- they can never fire in the same cycle.

        -- Activation along the loop back-edge.
        if backedge_valid = '1' and backedge_ready = '1' then
          if measuring then
            last_cyc := cycle;
            iters    := iters + 1;
          end if;
        end if;

        -- Control leaving the loop.
        if exit_valid = '1' and exit_ready = '1' then
          if measuring then
            report_window(iters, start_cyc, last_cyc);
          end if;
          measuring := false;
        end if;

        -- Activation from outside the loop (first entry or re-entry).
        if entry_valid = '1' and entry_ready = '1' then
          if measuring then
            report_window(iters, start_cyc, last_cyc);
          end if;
          measuring := true;
          start_cyc := cycle;
          last_cyc  := cycle;
          iters     := 1;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""
