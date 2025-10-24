from generators.support.unary import generate_unary


def generate_uitofp(name, params):
    latency = params["latency"]
    input_bitwidth = params["bitwidth"]

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
  float_value <= to_float(unsigned(ins));
  converted <= to_std_logic_vector(float_value);
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
      elsif (valid_buffer_ready) then
        q0 <= converted;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
        q4 <= q3;
      end if;
    end if;
  end process;
    """

    return generate_unary(
        name=name,
        handshake_op="uitofp",
        input_bitwidth=input_bitwidth,
        output_bitwidth=32,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
        latency=latency
    )
