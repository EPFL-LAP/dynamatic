

BB1: %9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
BB3: Operation: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
BB4: Operation: %42 = "handshake.join"(%35#0, %26#0) : (!handshake.control<>, !handshake.control<>) -> !handshake.control<>
BB4: peration: %43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
BB1: %6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
BB2: Operation: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
BB1:no_use %7 = "handshake.merge"(%26#0, %26#0, %26#0, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
BB1:no_use %8 = "handshake.merge"(%35#0, %35#0, %8, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>

