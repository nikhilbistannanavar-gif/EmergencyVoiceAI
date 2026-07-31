#ifndef FRONTEND_H
#define FRONTEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAMPLE_RATE    16000

#define FRAME_LENGTH   640
#define FRAME_STEP     320
#define FFT_LENGTH     1024
#define FFT_BINS       (FFT_LENGTH / 2 + 1)

#define MEL_BINS       64
#define TOTAL_FRAMES   49

void frontend_init(void);

void frontend_process(const int16_t *pcm);

const float *frontend_get_mel(void);

#ifdef __cplusplus
}
#endif

#endif