def generate_join(name, params):
    # Number of input ports
    size = params["size"]

    verilogjoin = f"""
// Module of join
module {name}(
  input [{size} - 1 : 0] ins_valid,
  input outs_ready,

  output reg  [{size} - 1 : 0] ins_ready = 0,
  output outs_valid
);
  
  assign outs_valid = &ins_valid; // AND of all the bits in ins_valid vector
  
  reg [{size} - 1 : 0] singleValid = 0;
  integer i, j;
  
  always @(*)begin
    for (i = 0; i < {size}; i = i + 1) begin
      singleValid[i] = 1;
      for (j = 0; j < {size}; j = j + 1)
        if (i != j)
          singleValid[i] = singleValid[i] & ins_valid[j];
    end
    
    for (i = 0; i < {size}; i = i + 1) begin
      ins_ready[i] = singleValid[i] & outs_ready;
    end
  end
  
endmodule

"""

    return verilogjoin
