# Writing Hls C Code for Dynamatic
Before passing your C kernel (function) to Dynamatic for compilation, it is important that you ensure it meets some guidelines. This document presents the said guidelines and some constraints that the user must follow to make their code suitable inputs for Dynamatic.

> [!NOTE]  
> These guidelines target the function to be compiled and not the `main` function of your program except for the `CALL_KERNEL`. Main is primarily useful for passing inputs for simulation and is not compiled by Dynamatic

## Summary
1. [Dynamatic header](#1-include-the-dynamatic-integration-header)
2. [`CALL_KERNEL` macro in `main`](#2-use-the-call_kernel-macro-in-the-main-function)
3. [Variable Types and Names in `main` Must Match Parameter Names in Kernel Declaration]()
4. [Inline functions called by the kernel](#3-do-not-call-functions-in-your-target-function)
5. [No recursive calls](#4-recursive-calls-are-not-supported)
6. [No pointers](#5-pointers--are-not-supported)
7. [No dynamic memory allocation](#6-dynamic-memory-allocation-is-not-supported)
8. [Pass global variables](#7-global-variables)
9. [No support for local array declarations](#8-local-array-declarations-are-not-supported)
10. [Data type support](#data-types-supported-by-dynamatic)

### 1. Include the Dynamatic Integration Header

To be able to compile in Dynamatic, your C files should include the `Integration.h` header that will be a starting point for accessing other relevant Dynamatic libraries at compile time.
```c
#include "dynamatic/Integration.h"
```
### 2. Use the CALL_KERNEL Macro in the `main` Function

Do not call the kernel function directly, use the `CALL_KERNEL` macro provided through Dynamatic's integration header. 
It does two things in the compiler flow:
- Dumps the argument passed to the kernel to files in sim/INPUT_VECTORS (for C/HDL cosimulation when the `simulate` command is ran).
- Dumps the argument passed to the kernel to a profiler to determine which loops are more important to be optimized using buffer placement.
```c
CALL_KERNEL(func, input_1, input_2, ... , input_n)
```
### 3. Match Variable Names and Types in `main` to the Parameter Declared as Kernel Inputs
For simulation purposes, the variables declared in the `main` function must have the same names and data types as the function parameters of your function under test. This makes it easy for the simulator to correctly identify and properly match parameters when passing them. For example:
```c
void loop_scaler(int arr[10], int scale_factor){
    ...
}; // function declaration

int main(){
    int arr[10]; // same name and type 
    int size;    // as in kernel declaration

    scale_factor = 50;
    // initialize arr[10] values

    CALL_KERNEL(loop_scaler, arr, scale_factor);
    return 0;
}
```

## Limitations
### 1. Do Not Call Functions in Your Target Function

The target function is the top level function to be implemented by Dynamatic. Dynamatic does not support calling other functions in the target kernel. Alternatively, you can use macros to implement any extra functionality before using them in your target kernel.
```c
#define increment(x) x+1; // macro for increment function

void loop(int x) {
    while (x<20) {
        increment(x); // macro
    }
}  // inlined with macro definition.
```
### 2. Recursive Calls Are Not Supported  
Like other HLS tools, Dynamatic does not support recursive function calls because:
- they are difficult to map to hardware
- have unpredictable depths and control flow
- unbounded execution
- the absence of call-stack in FPGA platforms would be too resource demanding to implement efficiently epecially without knowing the bounds ahead of time.  
An alternative would be to manually unroll recursive calls and replace them with loops where possible.  

### 3. Pointers  Are Not Supported 
Pointers should not be used.  `*(x + 1) = 4;` is invalid. Use regular indexing and fixed sized arrays if need be as shown below.
```c
int x[10]; // fixed sized
x[1] = 4; // non-pointer indexing
```
### 4. Dynamic Memory Allocation is Not Supported
Dynamic memory allocation is also not allowed because it's not deterministic enough to allow enough hardware resources to be allocated at compile time.  

### 5. Global Variables  
Dynamatic compiles the kernel code only. Any variables declared outside the kernel function will not be converted unless they are passed to the kernel. Global variables are no exception. You can pass global variables as parameters to your kernel or define them as macros to make your kernel simpler.

```c
#define scale_alternative (2)
int scale = 2; 

int scaler(int scale, int number) // scale is still passed as parameter
{ 
    return number * scale * scale_alternative;
}
```

### 6. Local Array Declarations are Not Supported  
Local array declaration in kernels is not yet supported by Dynamatic. Pass all arrays as parameters to your kernel.  

```c
void convolution(unsigned char input[HEIGHT][WIDTH], unsigned char output[HEIGHT][WIDTH]) {
    
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    int kernel_sum = 9;

    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += input[y + ky][x + kx] * kernel[ky + 1][kx + 1]; // one issue hear...non-affine apparently..
                                                                        // the kernel indexing is considered non-affine
                }
            }
            output[y][x] = sum / kernel_sum;
            printf("output[%d][%d] = %d\n", y, x, output[y][x]);
        }
    }
}
```
The above code will yield an error at compilation about array flattening. Pass it as a parameter to bypass the error:

```c
void convolution(int kernel[3][3], unsigned char input[HEIGHT][WIDTH], unsigned char output[HEIGHT][WIDTH])
```

## Data Types Supported by Dynamatic
These types are most crucial when dealing with function parameters. Some of the unsupported types may work on local variables without any compilation errors.
> [!NOTE]  
> Arrays of supported data types are also supported as function parameters

| buffer algorithm/data type | Supported
|---|---|
|unsigned | ✓ |
|int32_t / int16_t / int8_t|✓|
|uint32_t / uint16_t / uint8_t|✓|
|char / unsigned char | ✓|
|short|✓|
|float|✓|
|double|✓|
|long/long long/long double | x|
|uint64_t / int64_t | x |
__int128|x|

### Supported Operations
- Arithmetic operations: `+`, `-`, `*`, `/`, `++`, `--`.  
- Logical operations on `int`: `>`, `<`, `&&`, `||`, `!`, `^`

### Unsupported Operations
- Arithmetic operations: `%`
- Pointer operations: `*`, `&` (indexing is supported - `a[i]`)
- Most math functions excluding absolute value functions
- Logical operations can be used with variables of type `float` in C but the following are not yet supported in Dynamatic: `&&`, `||`, `!`, `^`.

> [!TIP]  
> Data type and operation related errors generally state explicitly that an operation or type is not supported. Kindly report those as bugs on our repository while we work on making more data types supported.

### Other C Constructs
#### Structs
`struct`s are currently not supported. Consider passing inputs individually rather than grouping with structs

#### Function Inlining
The `inline` keyword is not yet supported. Consider `#define` as an alternative for inlining blocks of code into your target function

#### Volatile
The `volatile` keyword is supported but has zero impact on the circuits generated. 
> [!WARNING]  
> Do not use on function parameters!  

Dynamatic is being refined over time and is yet to support certain constructs such as local array declarations in the target function which must rather be passed as inputs. If you encounter any issue in using Dynamatic, kindly report the bug on the github repository.

In the meantime, visit our [examples](../GettingStarted/Tutorials/Introduction/Examples.md) page to see an example of using Dynamatic.