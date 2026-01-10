from generators.support.arith_binary import generate_arith_binary


def generate_muli(name, params):
    bitwidth = params["bitwidth"]
    latency = params["latency"]
    lhs_actual_width = params["lhs_actual_width"]
    rhs_actual_width = params["rhs_actual_width"]

    assert(latency == 0 or latency == 4)

    if latency == 4:
        signals = f"""
      signal a_reg : std_logic_vector({lhs_actual_width} - 1 downto 0);
      signal b_reg : std_logic_vector({rhs_actual_width} - 1 downto 0);
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
          if (valid_buffer_ready = '1') then
            a_reg <= lhs({lhs_actual_width - 1} downto 0);
            b_reg <= rhs({rhs_actual_width - 1} downto 0);
            q0    <= mul;
            q1    <= q0;
            q2    <= q1;
          end if;
        end if;
      end process;

      result <= q2;
        """

        return generate_arith_binary(
            name=name,
            handshake_op="muli",
            bitwidth=bitwidth,
            signals=signals,
            body=body,
            extra_signals=params.get("extra_signals", None),
            latency=latency
        )
    elif latency == 0:
        signals = f"""
      signal a_trunc : std_logic_vector({lhs_actual_width} - 1 downto 0);
      signal b_trunc : std_logic_vector({rhs_actual_width} - 1 downto 0);
        """

        body = f"""

      a_trunc <= lhs({lhs_actual_width - 1} downto 0);
      b_trunc <= rhs({rhs_actual_width - 1} downto 0);
      
      result <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_trunc) * signed(b_trunc))), {bitwidth}));
        """

        return generate_arith_binary(
            name=name,
            handshake_op="muli",
            bitwidth=bitwidth,
            signals=signals,
            body=body,
            extra_signals=params.get("extra_signals", None),
            latency=0
        )
