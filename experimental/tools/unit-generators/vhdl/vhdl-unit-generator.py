from generators.handshake.cond_br import generate_cond_br

print(generate_cond_br("test_cond_br_spec", {
  "data_type": "!handshake.channel<i32>"
}))
print(generate_cond_br("test_cond_br_dataless_spec", {
  "data_type": "!handshake.control<[spec: i1]>"
}))
print(generate_cond_br("test_cond_br_spec", {
  "data_type": "!handshake.channel<i32, [spec: i1]>"
}))
