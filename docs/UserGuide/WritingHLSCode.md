[Table of Contents](../README.md)

# Writing Hls C Code for Dynamatic
Before passing your C kernel (function) to Dynamatic for compilation, it is important that you ensure it meets the following guidelines.

> [!NOTE]  
> These guidelines target the function to be compiled and not the `main` function of your program except for the `CALL_KERNEL`. Main is primarily useful for passing inputs for simulation and is not compiled by Dynamatic

## Summary
1. [Dynamatic header](#1-include-the-dynamatic-integration-header)
2. [`CALL_KERNEL` macro in `main`](#2-use-the-call_kernel-macro-in-the-main-function)
3. [Inline functions called by the kernel](#3-all-functions-called-by-your-target-function-must-be-inlined)
4. [No recursive calls](#4-recursive-calls)
5. [No pointers](#5-pointers)
6. [No dynamic memory allocation](#6-dynamic-memory-allocation)
7. [Pass global variables](#7-global-variables)
8. [No support for local array declarations](#8-local-array-declarations)
9. [Data type support](#data-type-support-for-dynamatic)

### **1. Include the Dynamatic integration header**

To be able to compile in Dynamatic, your C files should include the `Integration.h` header that will be a starting point for accessing other relevant Dynamatic libraries at compile time.
```
#include "dynamatic/Integration.h"
```
### **2. Use the CALL_KERNEL macro in the main function**

The `CALL_KERNEL` macro is available through Dynamatic's integration header. 
It does two things in the compiler flow:
- Dumps the argument passed to the kernel to files in sim/INPUT_VECTORS (for C/HDL cosimulation when the `simulate` command is ran).
- Dumps the argument passed to the kernel to a profiler to determine which loops are more important to be optimized using buffer placement.
```
CALL_KERNEL(func, input_1, input_2, ... , input_n)
```

### **3. All functions called by your target function must be inlined**

The target function is the top level function to be implemented by Dynamatic. 
```
#define increment(int x) x+1; // macro for increment function

void loop(x) {
    while (x<20) {
        increment(x); // macro
    }
}  // inlined with macro definition.
```
### **4. Recursive calls**  
Like other HLS tools, Dynamatic does not support recursive function calls because:
- they are difficult to map to hardware
- have unpredictable depths and control flow
- unbounded execution
- the absence of call-stack in FPGA platforms would be too resource demanding to implement efficiently epecially without knowing the bounds ahead of time.  

### **5. Pointers**  

Pointers should not be used.  `*(x + 1) = 4;` is invalid. Use regular indexing and fixed sized arrays if need be as shown below.
```
int x[10]; // fixed sized
x[1] = 4; // non-pointer indexing
```
### **6. Dynamic memory allocation**
Dynamic memory allocation is also disallowed because it's not deterministic enough to allow enough hardware resources to be allocated at compile time.  

### **7. Global variables**  
Pass global variables to functions as parameters else they will not be seen by Dynamatic at compilation and yield errors. See appropriate use below.

```
int scale = 2; 

int scaler(int scale, int number) // scale is still passed as parameter
{ 
    return number * scale;
}
```

### **8. Local Array declarations**  
Local array declaration are not yet supported. Pass all arrays as parameters.  

```
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

```
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

## Operations Supported by Dynamatic
|Type/Operation | int | float|int_example| float_example | working alternatives| comments|
|---|---|---|---|---|---|---|
|Incremenetation | ✓ | ✓ | `x++;`| `x++;` | `x += 1;`/`x += 0.1` |
|Decremenetation | ✓ | ✓ | `x--;`| `x--;` | `x -= 1;`/`x -= 0.1` |
|Comparison|✓ |✓| `x >= 1` | `x <= 0.45` | all comparison formats and operations are supported|
|AND|✓|x|`x && 2` | `x && 1.0f`(not supported just as in regular C)| `(x > 0.45f && x < 4.5f)` (works for compound conditions involving floats)| 
|OR and NOT| ✓ | x | `x \|\| y` | `x \|\| 0.45` (invalid) | see AND|
|Type Casting| ✓ |✓ | `(int)x` | `(float)x`| applies to all int and float/double types|
|Precompiled math functions| Only `abs`| only `fabsf`| `abs(x)` | `fabsf(x*2.0f)` | most precompiled C libraries are not supported. Consider coding custom functions for use with dynamatic as it requires requires explicit C|

> [!TIP]  
> Data type and operation related errors generally state explicitly that an operation or type is not supported. Kindly report those as bugs on our repository while we work on making more data types supported.

### Other C Constructs
#### Structs
`struct`s are currently not supported. Consider passing inputs individually rather than grouping with structs

#### Function Inlining
The `inline` keyword is not yet supported. Consider `#define` as an alternative for inlining blocks of code into your target function

#### Volatile
The `volatile` keyword is supported but has zero impact on the circuits generated. <span style="color:red;">Do not use on function parameters! </span>  

Dynamatic is being refined over time and is yet to support certain constructs such as local array declarations in the target function which must rather be passed as inputs. If you encounter any issue in using Dynamatic, kindly report the bug on the github repository.

In the meantime, visit our [examples](../GettingStarted/Tutorials/Introduction/Examples.md) page to see an example of using Dynamatic.