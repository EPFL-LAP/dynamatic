from generators.support.arith2 import generate_arith2


def generate_cmpf(name, params):
    impl = params["impl"]
    is_double = params["is_double"]
    extra_signals = params["extra_signals"]
    predicate = params["predicate"]

    modType = "cmpf"

    if impl == "flopoco":
        signals, body, bitwidth, latency = _get_flopoco(is_double, predicate)
    elif impl == "vivado":
       signals, body, bitwidth, latency = _get_vivado(is_double, predicate)

    dependencies = ""
    return generate_arith2(
          name,
          modType,
          bitwidth,
          signals,
          body,
          latency,
          dependencies,
          extra_signals
      )
    

##################################################
#                 Flopoco
##################################################

def _get_flopoco_expression_from_predicate(predicate):
  expressions = {
    "oeq": "not unordered and XeqY",
    "ogt": "not unordered and XgtY",
    "oge": "not unordered and XgeY",
    "olt": "not unordered and XltY",
    "ole": "not unordered and XleY",
    "one": "not unordered and not XeqY",
    "ueq": "unordered or XeqY",
    "ugt": "unordered or XgtY",
    "uge": "unordered or XgeY",
    "ult": "unordered or XltY",
    "ule": "unordered or XleY",
    "une": "unordered or not XeqY",
    "uno": "unordered"
  }
  if predicate not in expressions:
    raise ValueError(f"Unsupported flopoco predicate: {predicate}")

  return f"\"{expressions[predicate]}\""

def _flopoco_latency(is_double):
    return 1 if is_double else 0

def _bitwidth(is_double):
    return 64 if is_double else 32


def _get_flopoco(is_double, predicate):
    latency = _flopoco_latency(is_double)
    expression = _get_flopoco_expression_from_predicate(predicate)

    bitwidth = _bitwidth(is_double)

    signals = f"""
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;
  signal ip_lhs: std_logic_vector({bitwidth + 2} - 1 downto 0);
  signal ip_rhs: std_logic_vector({bitwidth + 2} - 1 downto 0);
  """
      
    body = f"""
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
        ce=> '1',
        X=> ip_lhs,
        Y=> ip_rhs,
        unordered=> unordered,
        XltY=> XltY,
        XeqY=> XeqY,
        XgtY=> XgtY,
        XleY=> XleY,
        XgeY=> XgeY);
  
  result(0) <= {expression};
"""

    return signals, body, bitwidth, latency

##################################################
#                      Vivado
##################################################

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


def _vivado_latency():
   return 2


def _get_vivado(is_double, predicate):
    if is_double:
        raise ValueError(f"Vivado cmpf does not support 64 bits")
        
    bitwidth = 32
    latency = _vivado_latency()
    predicate_code = _get_vivado_code_from_predicate(predicate)

    signals = f"""
  component cmpf_vitis_hls_wrapper is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 2;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 1
    );
    port (
      clk    : in  std_logic;
      reset  : in  std_logic;
      ce     : in  std_logic;
      din0   : in  std_logic_vector(din0_WIDTH - 1 downto 0);
      din1   : in  std_logic_vector(din1_WIDTH - 1 downto 0);
      opcode : in  std_logic_vector(4 downto 0);
      dout   : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal alu_opcode : std_logic_vector(4 downto 0);
"""
        
    body = f"""
  -- Predicate: {predicate}
  alu_opcode <= {predicate_code};
  array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component cmpf_vitis_hls_wrapper
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
    return signals, body, bitwidth, latency
