library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Synchronizes the input tokens (credits and operands).
-- Asserts valid only when all of them are present
-- - It has a handshake interface to each input,
-- - It has a single handshake interface for output (data are bundled together)

entity crush_sync is
  generic (
    NUM_OPERANDS : natural;
    DATA_WIDTH   : natural
  );
  port (
  -- Input data channels:
    ins        : in data_array (NUM_OPERANDS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector (NUM_OPERANDS - 1 downto 0);
    ins_ready  : out std_logic_vector (NUM_OPERANDS - 1 downto 0);

  -- Outputs (one valid bit):
    outs       : out data_array (NUM_OPERANDS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic; 
    outs_ready : in std_logic
       );
end crush_sync;

architecture arch of crush_sync is
begin
  outs <= ins;

  -- p_ins_ready: a process for assigning ready signal to input channels
  -- An input channel is ready if:
  -- 1. all other inputs have a valid token present.
  -- 2. the output channel is ready.
  p_ins_ready : process(ins_valid, outs_ready)
    variable allOtherInputsValid : std_logic_vector ((NUM_OPERANDS) - 1 downto 0);
  begin
    for i in 0 to NUM_OPERANDS - 1 loop
      allOtherInputsValid(i) := '1';
      for j in 0 to NUM_OPERANDS - 1 loop
        if (i /= j) then
          allOtherInputsValid(i) := (allOtherInputsValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to INPUTS-1 loop
      ins_ready(i) <=  (allOtherInputsValid(i) and outs_ready);
    end loop;
  end process;

  -- p_outs_valid : a process for assigning valid signal to the bundled output channel
  p_outs_valid : process (ins_valid)
    variable allInputChannelValid : std_logic;
  begin
    allInputChannelValid := '1';
    for i in 0 to NUM_OPERANDS - 1 loop
      allInputChannelValid := allInputChannelValid and ins_valid(i);
    end loop;
    outs_valid <= allInputChannelValid; 
  end process;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- crush_oh_mux : a non-elastic mux that operates on an one-hot sel signal.
entity crush_oh_mux is
  generic (
    MUX_WIDTH  : natural;
    DATA_WIDTH : natural
  );
  port (
    -- inputs
    ins        : in data_array (MUX_WIDTH - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    sel        : in std_logic_vector (MUX_WIDTH - 1 downto 0);
    outs       : out std_logic_vector (DATA_WIDTH - 1 downto 0)
  );
end crush_oh_mux;

architecture arch of crush_oh_mux is
begin
  p_sel : process (ins, sel)
    variable var_result : std_logic_vector (DATA_WIDTH - 1 downto 0);
  begin
    var_result := ins(MUX_WIDTH - 1);
    for i in sel'range loop
      if (sel(i) = '1') then
        var_result := ins(i);
      end loop;
    outs <= var_result;
  end process p_sel;
end crush_oh_mux;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- crush_oh_branch : an elastic branch that operates on an one-hot sel signal.
entity crush_oh_branch is
  generic (
    BRANCH_WIDTH : natural;
    DATA_WIDTH   : natural
          );
  port (
  -- data channel
  ins        : in std_logic_vector(DATA_WIDTH - 1 downto 0);
  ins_valid  : in std_logic;
  ins_ready  : out std_logic;

  -- sel channel
  sel        : in std_logic_vector (BRANCH_WIDTH - 1 downto 0);
  sel_valid  : in std_logic;
  sel_ready  : out std_logic;

  outs       : out data_array(BRANCH_WIDTH - 1 downto 0)(DATA_WIDTH - 1 downto 0)
  outs_valid : out std_logic_vector (BRANCH_WIDTH - 1 downto 0);
       );
end crush_oh_branch;

architecture arch of crush_oh_mux is
begin
  p_outs : process (ins)
  begin
    for i in outs'range loop
      outs(i) <= ins;
    end loop
  end p_outs;

  p_outs_valid : process (ins_valid, sel_valid, sel) 
    variable sel_valid_wide : std_logic_vector(BRANCH_WIDTH - 1 downto 0);
    variable ins_valid_wide : std_logic_vector(BRANCH_WIDTH - 1 downto 0);
  begin
    sel_valid_wide := (others => sel_valid);
    ins_valid_wide := (others => ins_valid);
    outs_valid <= ((sel and sel_valid_wide) and ins_valid_wide);
  end p_outs_valid;

  sel_ready <= ins_valid;
  ins_ready <= sel_valid;
end crush_oh_branch;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.math_real.all;
use work.customTypes.all;

entity credit is 
  generic (
            INPUTS        : integer;
            OUTPUTS       : integer;
            DATA_SIZE_IN  : integer;
            DATA_SIZE_OUT : integer;
            NUM_CREDITS       : integer
          );
  port (
  clk, rst      : in  std_logic;    
  pValidArray   : in  std_logic_vector (INPUTS - 1 downto 0);
  nReadyArray   : in  std_logic_vector (OUTPUTS - 1 downto 0);
  validArray    : out std_logic_vector (OUTPUTS - 1 downto 0);
  readyArray    : out std_logic_vector (INPUTS - 1 downto 0);
  -- one of the data input is a don't care, we simply not connecting anything
  -- to it.
  dataInArray   : in  data_array (INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0); 
  dataOutArray  : out data_array (OUTPUTS - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0)
);

begin
  assert INPUTS = OUTPUTS severity failure;
  assert INPUTS = 1 severity failure;
  assert NUM_CREDITS > 0 severity failure;

end credit;

architecture arch of credit is

  -- 1 + NUM_CREDITS since the bitwidth has to represent 0 to NUM_CREDITS, which is NUM_CREDITS + 1 numbers.
  constant counter_width : integer := integer(ceil(log2(real(1 + NUM_CREDITS))));

  signal output_transfer : std_logic := '0';
  signal input_transfer  : std_logic := '0';

  signal one: std_logic_vector (0 downto 0) := "1";
  signal zero: std_logic_vector (0 downto 0) := "0";

  -- initial value of the credit counter
  constant c_counter_init : unsigned (counter_width - 1 downto 0) := to_unsigned(NUM_CREDITS, counter_width);

  signal counter_reg  : unsigned (counter_width - 1 downto 0) := c_counter_init;
  signal valid_internal : std_logic := '0';
  signal ready_internal : std_logic := '0';

  signal full_reg : std_logic := '1';
  signal empty_reg : std_logic := '0';

begin

  output_transfer <= (valid_internal and nReadyArray(0));
  input_transfer <= (pValidArray(0) and ready_internal);

  p_update_counter : process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst) then
        -- the counter resets at the maximum number of credits
        counter_reg <= c_counter_init;
      elsif (input_transfer) and (not output_transfer) then
        -- whenever we return a credit but not add a new credit
        counter_reg <= counter_reg + 1;
      elsif (not input_transfer) and (output_transfer) then
        -- whenever we add a new credit but not return a credit 
        counter_reg <= counter_reg - 1;
      end if;
    end if;
  -- else (input_transfer) and (output_transfer), the counter_reg stays the same
  end process p_update_counter;

  validArray(0) <= valid_internal;
  readyArray(0) <= ready_internal;

  -- the credit counter should break combinational path in ready direction,
  -- the shared unit already breaks the path in valid direction
  ready_internal <= (not full_reg);
  -- valid_internal <= (not empty_reg) or pValidArray(0);
  valid_internal <= (not empty_reg);

  full_reg <= '1' when (counter_reg = c_counter_init) else '0';
  empty_reg <= '1' when (counter_reg = (counter_width -1 downto 0 => '0')) else '0';


end arch;
