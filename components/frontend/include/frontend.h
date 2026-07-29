#ifndef FRONTEND_H
#define FRONTEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_LENGTH   640
#define FRAME_STEP     320
#define FFT_LENGTH     1024
#define MEL_BINS       64

const float *frontend_get_mel(void);

void frontend_init(void);
void frontend_process(const int16_t *pcm);

#ifdef __cplusplus
}
#endif

#endif