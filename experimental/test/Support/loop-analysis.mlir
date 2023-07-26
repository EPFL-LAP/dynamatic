func.func @test1(%c0: i1, %c2: i1, %c3: i1) {
  cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4
  ^bb4:
    cf.cond_br %c2, ^bb2, ^bb5
  ^bb5:
    cf.cond_br %c2, ^bb1, ^bb6
  ^bb6:
    return
}