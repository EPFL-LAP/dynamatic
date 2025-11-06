library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity selector is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk		           : in std_logic;
    rst              : in std_logic;
    condition        : in std_logic_vector(0 downto 0);
    condition_valid  : in std_logic;
    trueValue        : in std_logic_vector(DATA_TYPE - 1 downto 0);
    trueValue_valid  : in std_logic;
    falseValue       : in std_logic_vector(DATA_TYPE - 1 downto 0);
    falseValue_valid : in std_logic;
    result_ready     : in std_logic;
    -- outputs
    result           : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid     : out std_logic;
    condition_ready  : out std_logic;
    trueValue_ready  : out std_logic;
    falseValue_ready : out std_logic
  );
end entity;

architecture arch of selector is
  -- One side can run ahead of the other for "discard_depth - 1" times
  -- NOTE: discard_depth > 1
  constant discard_depth : integer := 4;
  constant counter_width : positive := positive(ceil(log2(real(discard_depth))));

  -- number of tokens to discard for the true side
  signal num_token_to_discard_true  : std_logic_vector(counter_width - 1 downto 0) := (others => '0');
  -- number of tokens to discard for the false side
  signal num_token_to_discard_false : std_logic_vector(counter_width - 1 downto 0) := (others => '0');

  signal can_propagate_true : std_logic;
  signal can_propagate_false : std_logic;

  signal can_discard_true : std_logic;
  signal can_discard_false : std_logic;

  signal still_need_to_discard_true  : std_logic;
  signal still_need_to_discard_false : std_logic;
begin


  -- [START internal signal configuration]
  -- Select can discard more true if we are currently discarding a true or the counter hasn't reached the limit
  can_discard_true <= '1' when ((trueValue_valid = '1') or (unsigned(num_token_to_discard_true) < discard_depth)) else '0';
  -- Select can discard more false if we are currently discarding a true or the counter hasn't reached the limit
  can_discard_false <= '1' when ((falseValue_valid = '1') or (unsigned(num_token_to_discard_false) < discard_depth)) else '0';

  can_propagate_true <= '1' when 
    ((condition_valid = '1') and 
     (condition(0) = '1') and 
     (trueValue_valid = '1') and 
     (unsigned(num_token_to_discard_true) = 0) and 
     (can_discard_false = '1'))
  else '0';

  can_propagate_false <= '1' when 
    ((condition_valid = '1') and 
     (condition(0) = '0') and 
     (falseValue_valid = '1') and 
     (unsigned(num_token_to_discard_false) = 0) and 
     (can_discard_true = '1'))
  else '0';

  still_need_to_discard_true <= '1' when (unsigned(num_token_to_discard_true) > 0) else '0';
  still_need_to_discard_false <= '1' when (unsigned(num_token_to_discard_false) > 0) else '0';
  -- [END internal signal configuration]

  -- [START handshake configuration]
  -- true (false) token is valid to send if
  -- 1. there is nothing to discard for true (false)
  -- 2. the data is valid
  -- 3. the counter for false (true) is less than maximum
  result_valid <= can_propagate_true or can_propagate_false;
  result <= falseValue when (condition(0) = '0') else trueValue;

  -- Conditions:
  -- 1. "not trueValue_valid": prevent deadlocking.
  -- 2. "result_valid and result_ready": all three inputs are valid. In this case, take all of them and discard the unselected one.
  -- 3. "discard_true": false input and condition are passed without the true input in a previous cycle. Here we need to discard that one.
  trueValue_ready  <= (not trueValue_valid) or (result_valid and result_ready) or (still_need_to_discard_true); 
  falseValue_ready <= (not falseValue_valid) or (result_valid and result_ready) or (still_need_to_discard_false); 
  condition_ready  <= (not condition_valid) or (result_valid and result_ready); 
  -- [END handshake configuration]

  -- [START updating the discard counters]
  proc_counter_true : process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        num_token_to_discard_true <= (others => '0');
      else
        -- false token transfered without discarding true token
        if (result_valid and result_ready and (not trueValue_valid)) then
          num_token_to_discard_true <= std_logic_vector(unsigned(num_token_to_discard_true) + 1);
        -- discarding the true token while not taking any new false token
        elsif (still_need_to_discard_true and trueValue_valid) then
          num_token_to_discard_true <= std_logic_vector(unsigned(num_token_to_discard_true) - 1);
        end if;
      end if;
    end if;
  end process proc_counter_true;

  proc_counter_false : process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        num_token_to_discard_false <= (others => '0');
      else
        -- false token transfered without discarding true token
        if (result_valid and result_ready and (not falseValue_valid)) then
          num_token_to_discard_false <= std_logic_vector(unsigned(num_token_to_discard_false) + 1);
        -- discarding the false token while not taking any new false token
        elsif (still_need_to_discard_false and falseValue_valid) then
          num_token_to_discard_false <= std_logic_vector(unsigned(num_token_to_discard_false) - 1);
        end if;
      end if;
    end if;
  end process proc_counter_false;
  -- [END updating the discard counters]

end architecture;
