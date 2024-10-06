library ieee;
use ieee.std_logic_1164.all;

entity join is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins_valid  : in std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(SIZE - 1 downto 0)
  );
end join;

architecture arch of join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.and_n generic map(SIZE) port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(SIZE - 1 downto 0);
  begin
    for i in 0 to SIZE - 1 loop
      singlePValid(i) := '1';
      for j in 0 to SIZE - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to SIZE - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;

entity join_with_tag is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins_valid  : in std_logic_vector(SIZE - 1 downto 0);
    ins_spec_tag  : in std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready  : out std_logic_vector(SIZE - 1 downto 0)
  );
end join_with_tag;

architecture arch of join_with_tag is
  signal oneSpec : std_logic;
begin
  oneSpecOrGate : entity work.or_n generic map(SIZE) port map(ins_spec_tag, oneSpec);
  outs_spec_tag <= oneSpec;

  join_inner : entity work.join(arch) generic map(SIZE)
    port map(
      -- inputs
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      -- outputs
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );
end architecture;
