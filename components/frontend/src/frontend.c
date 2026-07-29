#include "frontend.h"

#include <stdio.h>
#include <string.h>
#include "dsps_fft2r.h"
#include "dsps_wind_hann.h"
#include <math.h>
#include "mel_filterbank.h"

static int16_t frame_buffer[FRAME_LENGTH];
static int16_t overlap_buffer[FRAME_STEP];

static float fft_buffer[FFT_LENGTH * 2];
static float magnitude[FFT_LENGTH / 2 + 1];
static float current_mel[MEL_BINS];
static float mel_spectrogram[49][MEL_BINS];
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
    // Previous 320 samples
    memcpy(frame_buffer,
           overlap_buffer,
           FRAME_STEP * sizeof(int16_t));

    // New 320 samples
    memcpy(frame_buffer + FRAME_STEP,
           pcm,
           FRAME_STEP * sizeof(int16_t));

    // Save overlap
    memcpy(overlap_buffer,
           frame_buffer + FRAME_STEP,
           FRAME_STEP * sizeof(int16_t));

    // Windowing
    for (int i = 0; i < FRAME_LENGTH; i++)
    {
        fft_buffer[2 * i] =
            ((float)frame_buffer[i] / 32768.0f) * window[i];

        fft_buffer[2 * i + 1] = 0.0f;
    }

    // Zero padding
    for (int i = FRAME_LENGTH; i < FFT_LENGTH; i++)
    {
        fft_buffer[2 * i] = 0.0f;
        fft_buffer[2 * i + 1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_buffer, FFT_LENGTH);
    dsps_bit_rev_fc32(fft_buffer, FFT_LENGTH);
    dsps_cplx2reC_fc32(fft_buffer, FFT_LENGTH);

    for (int i = 0; i <= FFT_LENGTH / 2; i++)
    {
        float real = fft_buffer[2 * i];
        float imag = fft_buffer[2 * i + 1];

        magnitude[i] = sqrtf(real * real + imag * imag);
    }

    mel_filterbank_compute(magnitude, current_mel);

    // Shift spectrogram
    for (int i = 0; i < 48; i++)
    {
        memcpy(mel_spectrogram[i],
               mel_spectrogram[i + 1],
               sizeof(float) * MEL_BINS);
    }

    memcpy(mel_spectrogram[48],
           current_mel,
           sizeof(float) * MEL_BINS);

    printf("Mel: ");

    for (int i = 0; i < 10; i++)
    {
        printf("%.3f ", current_mel[i]);
    }

    printf("\n");
}
const float *frontend_get_mel(void)
{
    return &mel_spectrogram[0][0];
}