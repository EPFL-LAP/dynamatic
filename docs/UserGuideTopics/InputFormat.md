[Home](../../README.md) <span>&ensp;</span> [Usage](Usage.md)<span>&ensp;</span> [Modification](AdvancedUsage.md)<span>&ensp;</span> [Advanced-Build](AdvancedBuild.md) <span>&ensp;</span>[Examples](Examples.md) <span>&ensp;</span>[Dependencies](Dependencies.md) <span>&ensp;</span>[Development](WorkInProgress.md)

[Data-Type-Support](DataTypeSupport.md)
# Dressing up your C/C++ code for Dynamatic
Before passing your C kernel (function) to Dynamatic for compilation, it is important that you ensure it meets the following guidelines.

> Note that these guidelines target the function to be compiled and not the `main` function of your program. Main is primarily useful for passing inputs for simulation and is not compiled by Dynamatic

## Summary
|Topic|Prescription|Example
|---|----|---|
|Dynamatic  Integration| include integration header| `#include "dynamatic/Integration.h`
| CALL_KERNEL | Use in main instead of regular function call| ```CALL_KERNEL(func, input_1, input_2,...,input_n);```
|Function Name|Make the same as C file name|loop_array.c > ```loop_array()...```|
|Nested functions| Inline before use in top level function| ```#define increment(x) (x+1)  /*space*/ void loop(x) { while (x<20) increment(x); }  ``` // inlined with macro definition. 
|Recursive calls|Not supported (consider alternatives such as manual unrolling where possible)|```factorial()``` // use loops instead of recursion
|Pointers| Not supported| ```x = *(arr+1);``` // use indexing instead of using pointer
|Dynamic memory allocation| Not supported | ```int *x = malloc(N * sizeof(int));``` // avoid/ Use static bound arrays and regular variable assignments
|Local arrays| Not supported yet. Pass such as inputs to your function| see example in the detailed section above
|Global variables| Pass as parameters|```int scale = 2; int scaler(int scale, int number) return number * scale;```|

<br/>
<br/>

## **1. Include the Dynamatic integration header**

To be able to compile in Dynamatic, your C files should include the `Integration.h` header that will be a starting point for accessing other relevant Dynamatic libraries at compile time.
```
#include "dynamatic/Integration.h"
```
## **2. Use the CALL_KERNEL macro in the main function**

The `CALL_KERNEL` macro calls the kernel while allowing us to automatically run code prior to and/or after the call. For example, this is used during C/VHDL co-verification to automatically write the C function's reference output to a file for later comparison with the generated VHDL design's output. It is especially useful if you plan on simulating your generated HDL code in Dynamatic.
```
CALL_KERNEL(func, input_1, input_2, ... , input_n)
```

## **4. All functions called by your target function must be inlined**

The target function is the top level function to be implemented by Dynamatic. 
```
#define increment(int x) x+1; // macro for increment function

void loop(x) {
    while (x<20) {
        increment(x); // macro
    }
}  // inlined with macro definition.
```
## **5. Recursive calls**  
Like other HLS tools, Dynamatic does not support recursive function calls because:
- they are difficult to map to hardware
- have unpredictable depths and control flow
- unbounded execution
- the absence of call-stack in FPGA platforms would be too resource demanding to implement efficiently epecially without knowing the bounds ahead of time.  

## **6. Pointers**  

Pointers should not be used.  `*(x + 1) = 4;` is invalid. Use regular indexing and fixed sized arrays if need be as shown below.
```
int x[10]; // fixed sized
x[1] = 4; // non-pointer indexing
```
## **7. Dynamic memory allocation**
Dynamic memory allocation is also disallowed because it's not deterministic enough to allow enough hardware resources to be allocated at compile time.
<br/>

## **8. Global variables**
Pass global variables to functions as parameters else they will not be seen by Dynamatic at compilation and yield errors. See appropriate use below.

```
int scale = 2; 

int scaler(int scale, int number) // scale is still passed as parameter
{ 
    return number * scale;
}
```

## **10. Local Array declarations**
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


Dynamatic is being refined over time and is yet to support certain constructs such as local array declarations in the target function which must rather be passed as inputs. If you encounter any issue in using Dynamatic, kindly report the bug on the github repository.

In the meantime, visit out [examples](Examples.md) page to see an example of using Dynamatic our [supported data types and operations](DataTypeSupport.md) page for information on the constructs currently supported by D    ynamatic.