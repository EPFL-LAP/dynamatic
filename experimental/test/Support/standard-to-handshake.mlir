// func.func @test1(%arg0 : i32, %arg1: i1) {
//   %cst = arith.constant 0 : i32
//   cf.cond_br %arg1, ^bb1, ^bb2(%arg0 : i32)
//   ^bb1:
//     cf.br ^bb2(%cst : i32)
//   ^bb2(%x : i32):
//     return 
// }

// func.func @test2 (%arg0 : i32, %arg1 : i1) {
// 	%cst = arith.constant 1 : i32
// 	%add = arith.addi %arg0, %cst : i32
// 	cf.br ^bb1(%add : i32)
// ^bb1(%x : i32):
// 	cf.cond_br %arg1, ^bb2(%x: i32), ^bb3
// ^bb2(%y : i32):
// 	%add2 = arith.addi %y, %cst : i32
//   %cst2 = arith.constant 38 : i32
//   %add3 = arith.addi %y, %cst2 : i32
// 	cf.br ^bb1(%add2 : i32)
// ^bb3:
// 	return
// }

// func.func @test3 () {
//   %b = arith.constant 0 : i1
//   %n = arith.constant 378 : i32
//   %n1 = arith.constant 121378 : i32
//   cf.cond_br %b, ^bb1, ^bb2
//   ^bb1:
//     %add = arith.addi %n, %n1 : i32
//     return
//   ^bb2:
//     return
// }


// func.func @test4 () {
// 	%b = arith.constant 0 : i1
//   %n = arith.constant 378 : i32
//   %n1 = arith.constant 121378 : i32
// 	cf.br ^bb1
// ^bb1:
// 	cf.br ^bb2(%n : i32)
// ^bb2(%n2 : i32):
//   %add = arith.addi %n, %n2 : i32
//   %add2 = arith.addi %n, %n2 : i32
// 	cf.cond_br %b, ^bb3(%add: i32), ^bb4
// ^bb3(%y : i32):
// 	cf.br ^bb2(%n : i32)
// ^bb4:
// 	return
// }

// func.func @test5() {
//   // Declare and initialize variables
//   %x = arith.constant 22 : i32
//   %b = arith.constant 1 : i1

//   // Conditional branch
//   cf.cond_br %b, ^if_true, ^if_false

//   // True branch
//   ^if_true:
//     %x_true = arith.constant 3 : i32
//     cf.br ^exit(%x_true : i32)

//   // False branch
//   ^if_false:
//     %x_prev = arith.addi %x, %x: i32
//     cf.br ^exit(%x_prev : i32)

//   // Exit block
//   ^exit(%z : i32):
//     %y = arith.constant 0 : i32
//     %y_res = arith.addi %y, %z : i32
//     return
// }

func.func @test6 () {
	%b = arith.constant 0 : i1
  %n = arith.constant 378 : i32
	cf.br ^bb1
^bb1:
  %n1 = arith.constant 121378 : i32
  %b1 = arith.constant 0 : i1
	cf.cond_br %b1, ^bb2(%n : i32), ^bb4
^bb2(%n2 : i32):
  %add = arith.addi %n, %n2 : i32
  %add2 = arith.addi %n, %n2 : i32
  %b2 = arith.constant 0 : i1
	cf.cond_br %b2, ^bb3(%add: i32), ^bb2(%add2: i32)
^bb3(%y : i32):
	cf.br ^bb1
^bb4:
	return
}
