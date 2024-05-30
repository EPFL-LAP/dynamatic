# Compiler Intrinsics

## Wait

> [!NOTE]
> This is a proposed design change; it is not implemented yet.

There are many scenarios in which one may want to explicitly specify synchronization constraints between variables at the source code level and have Dynamatic circuits honor these temporal relations on its corresponding dataflow channels. In particular, this proposal focuses on a particular type of synchronization we call *wait*. Our goal here is to introduce a standard way for users to enforce the *waiting* relation between two source-level variables and provide insights as to how the compiler will treat the associated compiler intrinsic, ultimately resulting in a dataflow circuit honoring the relation.

### Example

Consider the following `pop_and_wait` kernel.

```c
// Pop from a FIFO identified by an integer.
// Note that the function has no body, so it will be treated as an external
// function by Dynamatic (the user is ultimately expected to provide a circuit
// for it to connect to the Dynamatic-generated circuit).
int pop(int queueID);

// Pop first two elements from the FIFO and return their difference.
int pop_and_wait(int queueID) {
  int x = pop(queueID);
  int y = pop(queueID);
  return x - y;
}
```

If this were to be executed on a CPU with a software implementation of `pop`, the two `pop` calls would happen naturally in the order in which they were specified in the code, yielding a correct kernel result every time. However, the ordering of the calls is no longer guaranteed in the world of dataflow circuits. Both calls are in the same basic block and have no explicit data dependency between them, meaning that Dynamatic is free to "execute them" in any order according to the availability of their (identical) operand and to the internal queue popping logic. If the second `pop` executes before the first one, then the kernel will produce the negation of its expected result. For reference, the Handshake-level IR for this piece of code might look something like the following.

```mlir
handshake.func private @pop(channel<i32>, control) -> (i32, control)

handshake.func @pop_and_wait(%queueID: channel<i32>, %start: control) -> channel<i32> {
  %forkedQueueID:2  = fork [2] %queueID : channel<i32>
  %forkedStart:2    = fork [2] %start : channel<i32>
  %x, _             = instance @pop(%forkedQueueID#0, %forkedStart#0) : (channel<i32>, control) -> channel<i32>
  %y, _             = instance @pop(%forkedQueueID#1, %forkedStart#1) : (channel<i32>, control) -> channel<i32>
  %res              = arith.subi %x, %y : channel<i32>
  %output           = return %res : channel<i32>
  end %output : channel<i32>
}
```

### Creating a data dependency

We need a way, in the source code, to tell Dynamatic that the second `pop` should always happen after the first has produced its result. One way to enforce this is to create a "fake" data dependency that makes the second use of `queueID` *depend on* `x`, the result of the first `pop`. We propose to represent this using a family of `__wait` compiler intrinsics. The `pop_and_wait` kernel may be rewritten as follows.

```c
// Pop first two elements from the FIFO and return their difference.
int pop_and_wait(int queueID) {
  int x = pop(queueID);
  queueID = __wait_int(__int_to_token(x), queueID);
  int y = pop(queueID);
  return x - y;
}
```

`__wait_int` is a compiler intrinsic---a special function with a reserved name which Dynamatic will give special treatment too during compilation---that expresses the user's desire that its return value (here `queueID`) only becomes valid (in the dataflow sense) when both of its arguments become valid in the corresponding dataflow circuit. The return value's payload inherits the second arguments's (here `queueID`) payload. This effectively creates a data dependency between `x` and `queueID` in between the two `pop`s.

### Intrinsic prototypes

Supporting the family of `__wait` compiler intrinsics in source code amounts to adding the following function prototypes once to the main Dynamatic C header (that all kernels should include).

```c
// Opaque token type
typedef int Token;

// Family of __wait intrinsics for all supported types
char      __wait_char(Token waitFor, char data);
short     __wait_short(Token waitFor, short data);
int       __wait_int(Token waitFor, int data);
unsigned  __wait_unsigned(Token waitFor, unsigned data);
float     __wait_float(Token waitFor, float data);
double    __wait_double(Token waitFor, double data);

// Family of conversion functions to "Token" type 
Token     __char_to_token(char x);
Token     __short_to_token(short x);
Token     __int_to_token(int x);
Token     __unsigned_to_token(unsigned x);
Token     __float_to_token(float x);
Token     __double_to_token(double x);
```

The lack of support for function overloading in C forces us to have a collection of functions for all our supported types. The opaque `Token` type and its associated conversion functions (`__*_to_token`) allows us to have a unique type for the first argument of all `__wait` intrinsics, regardless of the payload's type. Without it we would have had to define a `__wait` variant for each type combination in its two arguments or resort to illegal C value casts that either do not compile or yield convoluted IRs. Each `__*_to_token` conversion function in the source code yield a single additional IR operation which can easily be removed during the compilation flow.

### Compiler support

Our example kernel would lower to a very simple IR at the cf (control flow) level.

```mlir
func.func @pop_and_wait(%queueID: i32) -> i32 {
  %x              = call @pop(%queueID) (i32) -> i32
  %firstPopToken  = call @__int_to_token(%firstPop) : (i32) -> i32
  %retQueueID     = call @__wait_int(%firstPopToken, %queueID) : (i32, i32) -> i32
  %y              = call @pop(%retQueueID) : (i32) -> i32
  %res            = arith.subi %x, %y : i32
  return %res : i32
}

func.func private @pop(i32) -> i32
func.func private @__wait_int(i32, i32) -> i32
func.func private @__int_to_token(i32) -> i32
```

During conversion to Handshake, Dynamatic would recognize the intrinsic functions via their name and yield appropriate IR constructs to implement the desired behavior.

```mlir
handshake.func @pop_and_wait(%queueID: channel<i32>, %start: control) -> (channel<i32>, control) {
  %forkedQueueID:2  = fork [2] %queueID : channel<i32>
  %forkedStart:3    = fork [2] %start : channel<i32>
  %x, _             = instance @pop(%forkedQueueID#0, %forkedStart#0) : (channel<i32>, control) -> channel<i32>
  %retQueueID       = wait %x, %forkedQueueID#1 : (channel<i32>, channel<i32>) -> channel<i32>
  %y, _             = instance @pop(%retQueueID, %forkedStart#1) : (channel<i32>, control) -> channel<i32>
  %res              = arith.subi %x, %y : channel<i32>
  %output           = return %res : channel<i32>
  end %output, %forkedStart#2 : channel<i32>, control
}

handshake.func private @pop(channel<i32>, control) -> (channel<i32>, control)
```

We hightlight two key intrinsic-related aspects of the `cf-to-handshake` conversion below.

1. The call to `__int_to_token` has completely disappeared from the IR (both as an operation inside the `@pop_and_wait` function and as an external function declaration). As mentionned previously, this family of conversion functions only serves the purpose of source-level type-checking, and do not map to any specific behavior in the resulting dataflow circuit.
2. The call to `__wait_int` was replaced by a new Handshake operation called `wait`, which implements the behavior we describe [above](#creating-a-data-dependency). All `__wait` variants can map to a single MLIR operation thanks to MLIR's support for custom per-operation type-checking semantics. Note that the `@__wait_int` external function declaration is no longer part of the IR either.
