#ifndef DYNAMATIC_SUPPORT_ESPRESSO_MAIN_H
#define DYNAMATIC_SUPPORT_ESPRESSO_MAIN_H

#define TRUE 1

#define FALSE 0

#include "dynamatic/Support/Espresso/espresso.h"

void getPLA(char *s, int option, pPLA *PLA, int out_type);
void runtime(void);
void init_runtime(void);

#ifdef __cplusplus
extern "C" {
#endif

char *run_espresso(char *s);

#ifdef __cplusplus
}
#endif

#endif // DYNAMATIC_SUPPORT_ESPRESSO_MAIN_H