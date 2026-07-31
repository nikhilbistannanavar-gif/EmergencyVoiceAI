#include "frontend.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "dsps_fft2r.h"
#include "dsps_wind_hann.h"

#include "mel_filterbank.h"

static int16_t frame_buffer[FRAME_LENGTH];
static int16_t overlap_buffer[FRAME_STEP];

static float fft_buffer[FFT_LENGTH * 2];
static float magnitude[FFT_BINS];

static float current_mel[MEL_BINS];
static float mel_spectrogram[TOTAL_FRAMES][MEL_BINS];

static float window[FRAME_LENGTH];

void frontend_init(void)
{
    memset(frame_buffer, 0, sizeof(frame_buffer));
    memset(overlap_buffer, 0, sizeof(overlap_buffer));
    memset(mel_spectrogram, 0, sizeof(mel_spectrogram));

    dsps_fft2r_init_fc32(NULL, FFT_LENGTH);
    dsps_wind_hann_f32(window, FRAME_LENGTH);

    mel_filterbank_init();

    printf("Frontend initialized\n");
}

void frontend_process(const int16_t *pcm)
{
    /* Build 640-sample frame */

    memcpy(frame_buffer,
           overlap_buffer,
           FRAME_STEP * sizeof(int16_t));

    memcpy(frame_buffer + FRAME_STEP,
           pcm,
           FRAME_STEP * sizeof(int16_t));

    memcpy(overlap_buffer,
           frame_buffer + FRAME_STEP,
           FRAME_STEP * sizeof(int16_t));

    /* Window */

    for (int i = 0; i < FRAME_LENGTH; i++)
    {
        fft_buffer[2 * i] =
            ((float)frame_buffer[i] / 32768.0f) * window[i];

        fft_buffer[2 * i + 1] = 0.0f;
    }

    /* Zero padding */

    for (int i = FRAME_LENGTH; i < FFT_LENGTH; i++)
    {
        fft_buffer[2 * i] = 0.0f;
        fft_buffer[2 * i + 1] = 0.0f;
    }

    /* FFT */

    dsps_fft2r_fc32(fft_buffer, FFT_LENGTH);
    dsps_bit_rev_fc32(fft_buffer, FFT_LENGTH);
    dsps_cplx2reC_fc32(fft_buffer, FFT_LENGTH);

    /* Magnitude */

    for (int i = 0; i < FFT_BINS; i++)
    {
        float real = fft_buffer[2 * i];
        float imag = fft_buffer[2 * i + 1];

        magnitude[i] = sqrtf(real * real + imag * imag);
    }

    /* TensorFlow-compatible Log-Mel */

    mel_filterbank_compute(magnitude, current_mel);

    /* Rolling 49-frame spectrogram */

    for (int i = 0; i < TOTAL_FRAMES - 1; i++)
    {
        memcpy(mel_spectrogram[i],
               mel_spectrogram[i + 1],
               sizeof(float) * MEL_BINS);
    }

    memcpy(mel_spectrogram[TOTAL_FRAMES - 1],
           current_mel,
           sizeof(float) * MEL_BINS);

    /* -------- DEBUG: Print complete 49x64 feature -------- */

    static int feature_printed = 0;
    static int frame_count = 0;

    frame_count++;

    if (!feature_printed && frame_count >= TOTAL_FRAMES)
    {
        feature_printed = 1;

        printf("\nBEGIN_FEATURE\n");

        const float *mel = frontend_get_mel();

        for (int i = 0; i < TOTAL_FRAMES * MEL_BINS; i++)
        {
            printf("%.6f\n", mel[i]);
        }

        printf("END_FEATURE\n");
    }
}   // <-- frontend_process() ENDS HERE


const float *frontend_get_mel(void)
{
    return &mel_spectrogram[0][0];
}