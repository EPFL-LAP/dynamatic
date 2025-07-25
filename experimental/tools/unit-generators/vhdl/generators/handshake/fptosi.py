from generators.support.arith1 import generate_arith1


def generate_fptosi(name, params):
    signals = f"""
  signal converted : std_logic_vector(32 - 1 downto 0);
  signal q0 : std_logic_vector(32 - 1 downto 0);
  signal q1 : std_logic_vector(32 - 1 downto 0);
  signal q2 : std_logic_vector(32 - 1 downto 0);
  signal q3 : std_logic_vector(32 - 1 downto 0);
  signal q4 : std_logic_vector(32 - 1 downto 0);
  signal float_value : float32;
    """

    body = f"""
  float_value <= to_float(ins);
  converted <= std_logic_vector(to_signed(float_value, 32));
  outs <= q4;

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (rst) then
        q0 <= (others => '0');
        q1 <= (others => '0');
        q2 <= (others => '0');
        q3 <= (others => '0');
        q4 <= (others => '0');
      elsif (one_slot_break_dv_ready) then
        q0 <= converted;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
        q4 <= q3;
      end if;
    end if;
  end process;
    """

    return generate_arith1(
        name=name,
        modType="fptosi",
        input_bitwidth=32,
        output_bitwidth=64,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
        latency=5
    )
