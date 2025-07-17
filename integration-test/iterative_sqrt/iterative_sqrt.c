/**#include "iterative_sqrt.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"


int iterative_sqrt(in_int_t A[10]) {
  int x= 5;
   return A[1]+x;
}


int main(void) {
  in_int_t A[10];
  //in_int_t n;

  //srand(13);
  //n = rand() % 100;
  //n = 25;
  //for (unsigned j = 0; j < 10; ++j)
    //  A[j] =  j;

  CALL_KERNEL(iterative_sqrt, A);
  return 0;
}*/



#include "iterative_sqrt.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"
/*
int iterative_sqrt(in_int_t n) {
  int low = 0, high = n, mid;
  while (low <= high) {
    // Divide by 2
    mid = ((low + high) >> 1);
    if (mid * mid == n) {
      return mid;
    }
    if (mid * mid < n) {
      low = mid + 1;
    }
    else {
      high = mid - 1;
    }
  }
  return high;
}*/
int iterative_sqrt(in_int_t n) {
 int low = 0, high = n, mid;
 while (low <= high) {
   // Divide by 2
   mid = ((low + high) >> 1);
   if (mid * mid == n) {
     //high = mid;
     //break;
     return mid;
   }
   else if (mid * mid < n) {
     low = mid + 1;
   }
   else {
     high = mid - 1;
   }
 }
 return high;
}

/*
int iterative_sqrt(in_int_t a,in_int_t b) {
  int val;
  if(a>30) val=a++;
  else if(a<10) val = a-2;
  else val = b+4;
while(val>10)
  if(b<7)
    val--;
  else if ((b<10)){val-=2;}
  else if ((b<5)){val+=1;}
  else if ((b<3)){val+=2;}
  //else if ((b<2)){val-=1;}
  else val = a>>1;
  /
  for(int i=0;i<3;i++){
    n = n*2;
  }
  

  return val;
}*/

int main(void) {
  in_int_t A[10];
  in_int_t n,a,b;
  
  srand(13);
  //n = rand() % 100;
  n=15;
  a=15;b=14;
  for (unsigned j = 5; j < 15; ++j)
      A[j-5] =  j;
  CALL_KERNEL(iterative_sqrt, n);
  return 0;
}