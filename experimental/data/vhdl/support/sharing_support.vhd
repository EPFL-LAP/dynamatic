library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Simple request handling: it just takes whatever is more prioritized
-- > request : a vector of requests encoded as 0/1
-- > grant : a one-hot encoded signal indicates which request has been granted
-- The convention is that LSBs are more prioritized than MSBs

-- for example, imagine you have 2 request lines have value "11", let's compute
-- the corresponding grant signal
-- => ("11" and (not ("11" - 1)))
-- => ("11" and (not ("11" + "11")))
-- => ("11" and (not ("10")))
-- => ("11" and "01") => "01", this indicates the LSB is granted.

entity bitscan is
  generic (
            size : natural
          );
  port (
         request : in std_logic_vector(size - 1 downto 0);
         grant : out std_logic_vector(size - 1 downto 0)
       );
begin
end bitscan;

architecture arch of bitscan is
begin
  p_bitscan : process(request)
  begin
    grant <= std_logic_vector(unsigned(request) and (not (unsigned(request) - 1)));
  end process p_bitscan;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Synchronizes the input tokens (credits and operands). Asserts valid only when all of them are present
-- > It has a handshake interface to each input,
-- > It has a single handshake interface for output (data are bundled together)

entity crush_sync is
  generic (
            NUM_OPERANDS : natural;
            DATA_WIDTH : natural
          );
  port (
    -- Input data channels:
         ins : in data_array (NUM_OPERANDS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
         ins_valid : in std_logic_vector (NUM_OPERANDS - 1 downto 0);
         ins_ready : out std_logic_vector (NUM_OPERANDS - 1 downto 0);

    -- Outputs (one valid bit):
         outs : out data_array (NUM_OPERANDS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
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
    variable all_other_inputs_valid : std_logic_vector ((NUM_OPERANDS) - 1 downto 0);
  begin
    for i in 0 to NUM_OPERANDS - 1 loop
      all_other_inputs_valid(i) := '1';
      for j in 0 to NUM_OPERANDS - 1 loop
        if (i /= j) then
          all_other_inputs_valid(i) := (all_other_inputs_valid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to NUM_OPERANDS-1 loop
      ins_ready(i) <= (all_other_inputs_valid(i) and outs_ready);
    end loop;
  end process p_ins_ready;

  -- p_outs_valid : a process for assigning valid signal to the bundled output channel
  p_outs_valid : process (ins_valid)
    variable all_input_channels_valid : std_logic;
  begin
    all_input_channels_valid := '1';
    for i in 0 to NUM_OPERANDS - 1 loop
      all_input_channels_valid := all_input_channels_valid and ins_valid(i);
    end loop;
    outs_valid <= all_input_channels_valid; 
  end process p_outs_valid;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- crush_oh_mux : a non-elastic mux that operates on an one-hot sel signal.
entity crush_oh_mux is
  generic (
            MUX_WIDTH : natural;
            DATA_WIDTH : natural
          );
  port (
    -- inputs
         ins : in data_array (MUX_WIDTH - 1 downto 0)(DATA_WIDTH - 1 downto 0);
         sel : in std_logic_vector (MUX_WIDTH - 1 downto 0);
         outs : out std_logic_vector (DATA_WIDTH - 1 downto 0)
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
      end if;
    end loop;
    outs <= var_result;
  end process p_sel;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- crush_oh_branch : an elastic branch that operates on an one-hot sel signal.
entity crush_oh_branch is
  generic (
            BRANCH_WIDTH : natural;
            DATA_WIDTH : natural
          );
  port (
  -- data channel
         ins : in std_logic_vector(DATA_WIDTH - 1 downto 0);
         ins_valid : in std_logic;
         ins_ready : out std_logic;

  -- sel channel
         sel : in std_logic_vector (BRANCH_WIDTH - 1 downto 0);
         sel_valid : in std_logic;
         sel_ready : out std_logic;

          -- output channel
         outs : out data_array(BRANCH_WIDTH - 1 downto 0)(DATA_WIDTH - 1 downto 0);
         outs_valid : out std_logic_vector (BRANCH_WIDTH - 1 downto 0);
         outs_ready : in std_logic_vector (BRANCH_WIDTH - 1 downto 0)
);
end crush_oh_branch;

architecture arch of crush_oh_branch is
begin
  p_outs : process (ins)
  begin
    for i in outs'range loop
      outs(i) <= ins;
    end loop;
  end process p_outs;

  p_outs_valid : process (ins_valid, sel_valid, sel) 
    variable sel_valid_wide : std_logic_vector(BRANCH_WIDTH - 1 downto 0);
    variable ins_valid_wide : std_logic_vector(BRANCH_WIDTH - 1 downto 0);
  begin
    sel_valid_wide := (others => sel_valid);
    ins_valid_wide := (others => ins_valid);
    outs_valid <= ((sel and sel_valid_wide) and ins_valid_wide);
  end process p_outs_valid;

  p_ins_ready : process (outs_ready, ins_valid, sel_valid, sel)
    variable output_stalled : std_logic;
  begin
    output_stalled := '0';
    for i in outs_valid'range loop
      if sel(i) = '1'then
        output_stalled := output_stalled or (not outs_ready(i));
      end if;
    end loop;
    sel_ready <= ins_valid and (not output_stalled);
    ins_ready <= sel_valid and (not output_stalled);
  end process p_ins_ready;
end arch;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.math_real.all;
use work.types.all;

entity crush_credit_dataless is 
  generic (
            NUM_CREDITS : integer
          );
  port (
  clk, rst : in  std_logic;    
  -- input channels
  ins_valid : in  std_logic;
  ins_ready : out std_logic;
  -- output channels
  outs_valid : out std_logic;
  outs_ready : in  std_logic
);
end crush_credit_dataless;

architecture arch of crush_credit_dataless is

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

  output_transfer <= (valid_internal and outs_ready);
  input_transfer <= (ins_valid and ready_internal);

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

  outs_valid <= valid_internal;
  ins_ready <= ready_internal;

  -- the credit counter should break combinational path in ready direction,
  -- the shared unit already breaks the path in valid direction
  ready_internal <= (not full_reg);
  valid_internal <= (not empty_reg);

  full_reg <= '1' when (counter_reg = c_counter_init) else '0';
  empty_reg <= '1' when (counter_reg = (counter_width -1 downto 0 => '0')) else '0';

end arch;
