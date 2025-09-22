library ieee;
use ieee.std_logic_1164.all;

entity tb_join is
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
end entity;

architecture arch of tb_join is
  signal allValid  : std_logic;
  constant allOnes : std_logic_vector(SIZE - 1 downto 0) := (others => '1');
begin
  outs_valid <= '1' when ins_valid = allOnes else '0';

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
