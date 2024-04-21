#include "float_basic.h"

#include "dynamatic/Integration.h"
#include <stdlib.h>

void float_basic(in_float_t A[30][30], in_float_t B[30][30], out_float_t y[30] , inout_float_t x[30])
{
  int i, j;

 for (i = 0; i < 30; i++)
    {
      //float t_tmp = 0;
      float t_y = 0;

      for (j = 0; j < 30; j++){
		float t_x = x[j];
		if(A[i][j] <= B[i][j])
			t_y += A[i][j] / t_x;
		else
			t_y -= B[i][j] * t_x;
		}
      y[i] = t_y;
    }
}


int main(void){
	  //in_float_t alpha;
	  //in_float_t beta;
	  in_float_t A[30][30];
	  in_float_t B[30][30];
	  //out_float_t tmp[30];
	  out_float_t y[30];
	  inout_float_t x[30];
    
    //alpha = 1;
    //beta = 1;
    	for(int  j = 0; j  < 30; ++j){
    		//tmp[j] = 1;
			x[j] = (in_float_t)rand()/RAND_MAX;
			//y[j] = rand()%2;
    	    for(int k = 0; k < 30; ++k){
			      A[j][k] = (in_float_t)rand()/RAND_MAX*10;
			      B[j][k] = (in_float_t)rand()/RAND_MAX*10;
			}
		}

	CALL_KERNEL(float_basic, A, B, y, x);
	return 0;
}




