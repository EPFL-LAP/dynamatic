from generators.support.arith1 import generate_arith1


def generate_maximumf(name, params):
    signals = f"""
  component my_maxf is
    port (
      ap_clk    : in  std_logic;
      ap_rst    : in  std_logic;
      a         : in  std_logic_vector (32 - 1 downto 0);
      b         : in  std_logic_vector (32 - 1 downto 0);
      ap_return : out std_logic_vector (32 - 1 downto 0));
  end component;
    """

    body = f"""
  my_maxf_U1 : component my_maxf
    port map(
      ap_clk    => clk,
      ap_rst    => rst,
      a         => lhs,
      b         => rhs,
      ap_return => result
    );
    """

    return generate_arith1(
        name=name,
        modType="maximumf",
        bitwidth=32,
        signals=signals,
        body=body,
        extra_signals=params.get("extra_signals", None),
        latency=1
    )
