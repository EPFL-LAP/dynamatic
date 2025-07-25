from generators.support.arith2 import generate_arith2


def generate_muli(name, params):
    bitwidth = params["bitwidth"]

    signals = f"""
  signal a_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal b_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal q0    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q1    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q2    : std_logic_vector({bitwidth} - 1 downto 0);
  signal mul   : std_logic_vector({bitwidth} - 1 downto 0);
    """

    body = f"""
  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), {bitwidth}));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (one_slot_break_dv_ready = '1') then
        a_reg <= lhs;
        b_reg <= rhs;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  result <= q2;
    """

    return generate_arith2(
        name=name,
        modType="muli",
        bitwidth=bitwidth,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
        latency=4
    )
