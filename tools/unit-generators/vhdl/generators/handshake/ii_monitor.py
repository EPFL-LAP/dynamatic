from generators.support.utils import data, ExtraSignals


def _extra_signal_ports(channel_name: str,
                        extra_signals: ExtraSignals) -> list[str]:
    return [
        f"{channel_name}_{signal_name} : in std_logic_vector({signal_width} - 1 downto 0)"
        for signal_name, signal_width in extra_signals.items()
    ]


def _channel_ports(prefix: str, bitwidths: list[int],
                   extra_signals: list[ExtraSignals]) -> list[str]:
    """Ports of the observed channels named '<prefix>0', '<prefix>1', ..."""
    ports = []
    for idx, (bitwidth, extra) in enumerate(zip(bitwidths, extra_signals)):
        ports += [
            data(f"{prefix}{idx}       : in std_logic_vector({bitwidth} - 1 downto 0)",
                 bitwidth),
            f"{prefix}{idx}_valid : in std_logic",
            f"{prefix}{idx}_ready : in std_logic",
            *_extra_signal_ports(f"{prefix}{idx}", extra),
        ]
    return [port for port in ports if port]


def _sel_wiring(num_sels: int, init_selects: list[int]) -> str:
    """
    Concurrent assignments gathering the individual select ports into indexable signals so that the process can loop
    over the channels instead of unrolling them.
    """

    lines = []
    for idx in range(num_sels):
        init_test = " or ".join(
            f"unsigned(sel{idx}) = {value}" for value in init_selects)
        lines += [
            f"  sel_fire({idx}) <= sel{idx}_valid and sel{idx}_ready;",
            f"  sel_init({idx}) <= '1' when {init_test} else '0';",
        ]
    return "\n".join(lines)


def generate_ii_monitor(name, params):
    # Nesting depth of the measured loop (1 for a top-level loop) and deepest
    # depth reachable from it through its own descendants (equal to
    # loop_depth when the loop is innermost).
    loop_depth = params["loop_depth"]
    loop_max_depth = params["loop_max_depth"]
    # The select values that mark the first iteration of an activation: the
    # indices of the loop header's control merge inputs fed from outside the
    # loop.
    init_selects = params["init_selects"]
    # One entry per observed select channel; their number is what tells the
    # monitor how many of them to observe. The monitor ignores every
    # channel's extra signals (e.g. a "spec" signal under speculation), but
    # must declare matching ports so that its instantiation in the parent
    # module is well-formed.
    sel_bitwidths = params["sel_bitwidths"]
    sel_extra_signals = params["sel_extra_signals"]

    num_sels = len(sel_bitwidths)

    # Assemble the port list. All ports are inputs so the monitor never drives
    # the circuit (it only taps existing wires).
    ports = [
        "clk            : in std_logic",
        "rst            : in std_logic",
        # Select channels of the loop's header muxes.
        *_channel_ports("sel", sel_bitwidths, sel_extra_signals),
    ]
    port_decls = ";\n    ".join(ports)

    return f'''
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- II monitor: simulation-only component that reports the initiation of every
-- iteration of a loop as it happens, by observing the select channels of the
-- loop's header muxes.
--
-- All ports are inputs so that the monitor passively reads existing signals
-- without driving any of them, avoiding multiple-driver conflicts with the
-- actual circuit.
--
-- Each header mux carries a value from one iteration of the loop to the
-- next, and its select is consumed the very cycle the mux fires: together
-- with the initial value from outside the loop on an activation's first
-- iteration, and with the value the previous iteration fed back on every
-- later one. Every mux consumes the same select-token sequence (the header
-- merge's index, forked), so the k-th fire of every channel belongs to
-- iteration k: an iteration has been taken in once the *slowest* channel has
-- consumed it, i.e. whenever the smallest of the per-channel fire counts
-- advances.
-- 
-- At each such completion the monitor prints one line: the fixed
-- "II_INSTRUMENT: " prefix (what a reader locates the reports by) followed
-- by a JSON object,
--
--   II_INSTRUMENT: {{"loop": "<path>", "depth": <d>, "max_depth": <m>,
--                   "iter": <i>, "interval": <n>}}
--
-- where "iter" is the iteration's index within its activation and
-- "interval" the cycles since the previous iteration was taken in (null on
-- the run's very first line). The aggregation policy (means, average, etc.)
-- remains open to the consumer.
entity {name} is
  port (
    {port_decls}
  );
end entity;

architecture arch of {name} is
  type int_array is array (natural range <>) of integer;

  -- Per channel: whether it fires this cycle, and whether the value it
  -- consumes marks an activation's first iteration.
  signal sel_fire : std_logic_vector(0 to {num_sels} - 1);
  signal sel_init : std_logic_vector(0 to {num_sels} - 1);
begin
{_sel_wiring(num_sels, init_selects)}

  process (clk)
    variable cycle                 : integer := 0;
    -- Tokens each select channel has consumed so far.
    variable sel_fire_count        : int_array(0 to {num_sels} - 1) := (others => 0);
    -- Iterations taken in so far (the smallest of the fire counts), the
    -- cycle the latest one was taken in at, and its index within its
    -- activation.
    variable joined_iterations     : integer := 0;
    variable last_join_cycle       : integer := 0;
    variable iter_index            : integer := 0;
    -- Scratch, per cycle: the fire counts' new minimum.
    variable new_joined_iterations : integer;
    variable is_first_iteration    : boolean;

    -- Renders the interval since the previous iteration was taken in, which
    -- does not exist for the run's first.
    function interval_image(cycle, last_join_cycle : integer; first : boolean)
        return string is
    begin
      if first then
        return "null";
      end if;
      return integer'image(cycle - last_join_cycle);
    end function;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        cycle             := 0;
        sel_fire_count    := (others => 0);
        joined_iterations := 0;
        last_join_cycle   := 0;
        iter_index        := 0;
      else
        cycle := cycle + 1;

        for idx in sel_fire_count'range loop
          if sel_fire(idx) = '1' then
            sel_fire_count(idx) := sel_fire_count(idx) + 1;
          end if;
        end loop;

        -- An iteration is taken in once every channel has consumed it. The
        -- minimum advances by at most one per cycle (each channel fires at
        -- most once), so this reports each iteration exactly once.
        new_joined_iterations := sel_fire_count(0);
        for idx in sel_fire_count'range loop
          if sel_fire_count(idx) < new_joined_iterations then
            new_joined_iterations := sel_fire_count(idx);
          end if;
        end loop;

        if new_joined_iterations > joined_iterations then
          -- A channel whose fire just completed the iteration carries its
          -- select value this very cycle: whether it is an activation's
          -- first.
          is_first_iteration := false;
          for idx in sel_fire'range loop
            if sel_fire(idx) = '1' and
               sel_fire_count(idx) = new_joined_iterations then
              is_first_iteration := sel_init(idx) = '1';
            end if;
          end loop;
          if is_first_iteration then
            iter_index := 0;
          else
            iter_index := iter_index + 1;
          end if;

          report "II_INSTRUMENT: {{""loop"": """ & {name}'path_name &
                 """, ""depth"": {loop_depth}" &
                 ", ""max_depth"": {loop_max_depth}" &
                 ", ""iter"": " & integer'image(iter_index) &
                 ", ""interval"": " &
                 interval_image(cycle, last_join_cycle,
                                joined_iterations = 0) & "}}"
                 severity note;

          last_join_cycle   := cycle;
          joined_iterations := new_joined_iterations;
        end if;
      end if;
    end if;
  end process;
end architecture;
'''
