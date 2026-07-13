from generators.support.arith_binary import generate_arith_binary
from generators.support.utils import VIVADO_IMPL, FLOPOCO_IMPL


def generate_cmpf(name, params):
    predicate = params["predicate"]
    impl = params["fpu_impl"]
    latency = params["latency"]

    # only used by flopoco
    is_double = params.get("is_double", None)

    if impl == FLOPOCO_IMPL:
        bitwidth = 64 if is_double else 32
        if is_double is None:
            raise ValueError(f"is_double was missing for generating a flopoco cmpf")

        signals = _get_flopoco_signals(bitwidth, latency)
        body = _get_flopoco_body(bitwidth, predicate, latency)
    elif impl == VIVADO_IMPL:
        signals = _get_vivado_signals()
        body = _get_vivado_body(predicate)
        bitwidth = 32
    else:
        raise ValueError(f"Invalid fpu implementation on cmpf: {impl}")

    return generate_arith_binary(
        name=name,
        handshake_op="cmpf",
        input_bitwidth=bitwidth,
        output_bitwidth=1,
        signals=signals,
        body=body,
        latency=latency,
        extra_signals=params.get("extra_signals", None)
    )


##################################################
#                 Flopoco
##################################################


def _get_flopoco_signals(bitwidth, latency):
    # For the pipelined (latency > 0) core, the 'unordered' and 'XeqY' outputs
    # are combinational while the inequality outputs are pipelined (see the
    # comment in '_get_flopoco_body'). We delay the former by one cycle so
    # that every output feeding the result expression is valid at the same cycle.
    delayed_signals = ""
    if latency > 0:
        delayed_signals = """
  signal unordered_delayed : std_logic;
  signal XeqY_delayed : std_logic;"""

    return f"""
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;
  signal ip_lhs: std_logic_vector({bitwidth + 2} - 1 downto 0);
  signal ip_rhs: std_logic_vector({bitwidth + 2} - 1 downto 0);{delayed_signals}
  """


def _get_flopoco_body(bitwidth, predicate, latency):
    # A pipelined core (latency > 0, e.g. the 64-bit one) must have its clock
    # enable driven by 'valid_buffer_ready' so the pipeline freezes on
    # downstream stalls; a combinational core (latency == 0) has no valid buffer
    # and ties 'ce' high.
    #
    # The pipelined core is also internally inconsistent: 'XltY/XgtY/XleY/XgeY'
    # are registered (1-cycle) but 'unordered' and 'XeqY' are combinational
    # (0-cycle). We delay 'unordered' and 'XeqY' by one cycle to make every
    # output identical in latency.
    #
    # This is arguably a bug in flopoco and/or something that could be modeled
    # in components.json in the future.
    # TODO: Double check this/fix when regenerating the FloPoCo IP!
    if latency > 0:
        clock_enable = "valid_buffer_ready"
        unordered_sig, xeqy_sig = "unordered_delayed", "XeqY_delayed"
        align_process = """
  align_delayed : process (clk) is
  begin
    if rising_edge(clk) then
      if valid_buffer_ready = '1' then
        unordered_delayed <= unordered;
        XeqY_delayed <= XeqY;
      end if;
    end if;
  end process;
"""
    else:
        clock_enable = "'1'"
        unordered_sig, xeqy_sig = "unordered", "XeqY"
        align_process = ""

    expression = _get_flopoco_expression_from_predicate(
        predicate, unordered_sig, xeqy_sig)
    return f"""
  ieee2nfloat_0: entity work.InputIEEE_{bitwidth}bit(arch)
    port map(
        --input
        X=> lhs,
        --output
        R=> ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_{bitwidth}bit(arch)
    port map(
        --input
        X=> rhs,
        --output
        R=> ip_rhs
    );
  operator: entity work.FPComparator_{bitwidth}bit(arch)
  port map (clk=> clk,
        ce=> {clock_enable},
        X=> ip_lhs,
        Y=> ip_rhs,
        unordered=> unordered,
        XltY=> XltY,
        XeqY=> XeqY,
        XgtY=> XgtY,
        XleY=> XleY,
        XgeY=> XgeY);
{align_process}
  result(0) <= {expression};
  """


def _get_flopoco_expression_from_predicate(predicate, unordered_sig, xeqy_sig):
    expressions = {
        "oeq": f"not {unordered_sig} and {xeqy_sig}",
        "ogt": f"not {unordered_sig} and XgtY",
        "oge": f"not {unordered_sig} and XgeY",
        "olt": f"not {unordered_sig} and XltY",
        "ole": f"not {unordered_sig} and XleY",
        "one": f"not {unordered_sig} and not {xeqy_sig}",
        "ord": f"not {unordered_sig}",
        "ueq": f"{unordered_sig} or {xeqy_sig}",
        "ugt": f"{unordered_sig} or XgtY",
        "uge": f"{unordered_sig} or XgeY",
        "ult": f"{unordered_sig} or XltY",
        "ule": f"{unordered_sig} or XleY",
        "une": f"{unordered_sig} or not {xeqy_sig}",
        "uno": f"{unordered_sig}",
    }
    if predicate not in expressions:
        raise ValueError(f"Unsupported flopoco predicate: {predicate}")

    return f"{expressions[predicate]}"


##################################################
#                      Vivado
##################################################


def _get_vivado_signals():
    return f"""
  signal alu_opcode : std_logic_vector(4 downto 0);
"""


def _get_vivado_body(predicate):
    predicate_code = _get_vivado_code_from_predicate(predicate)
    return f"""
      -- Predicate: {predicate}
      alu_opcode <= {predicate_code};
      array_RAM_fcmp_32ns_32ns_1_2_1_u1 : entity work.cmpf_vitis_hls_wrapper
        generic map(
          ID         => 1,
          NUM_STAGE  => 2,
          din0_WIDTH => 32,
          din1_WIDTH => 32,
          dout_WIDTH => 1)
        port map(
          clk     => clk,
          reset   => rst,
          din0    => lhs,
          din1    => rhs,
          ce      => oehb_ready,
          opcode  => alu_opcode,
          dout(0) => result(0)
        );
    """


def _get_vivado_code_from_predicate(predicate):
    codes = {
        "oeq": "00001",
        "ogt": "00010",
        "oge": "00011",
        "olt": "00100",
        "ole": "00101",
        "one": "00110",
        "uno": "01000",
    }
    if predicate not in codes:
        raise ValueError(f"Unsupported vivado predicate: {predicate}")

    return f"\"{codes[predicate]}\""
