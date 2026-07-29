#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include <math.h>
#include "frontend.h"
#include "../model/model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#define I2S_BCLK_GPIO      GPIO_NUM_4
#define I2S_WS_GPIO        GPIO_NUM_5
#define I2S_DATA_GPIO      GPIO_NUM_7
static i2s_chan_handle_t rx_chan = NULL;
constexpr int kTensorArenaSize = 110 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
static void i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0,
        I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),

    .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_MONO),

    .gpio_cfg = {
        .mclk = I2S_GPIO_UNUSED,
        .bclk = I2S_BCLK_GPIO,
        .ws = I2S_WS_GPIO,
        .dout = I2S_GPIO_UNUSED,
        .din = I2S_DATA_GPIO,
        .invert_flags = {
            .mclk_inv = false,
            .bclk_inv = false,
            .ws_inv = false,
        },
    },
};

std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

static int32_t i2s_buffer[FRAME_STEP];

static void read_microphone(int16_t *pcm)
{
    size_t bytes_read = 0;

    ESP_ERROR_CHECK(
        i2s_channel_read(
            rx_chan,
            i2s_buffer,
            sizeof(i2s_buffer),
            &bytes_read,
            portMAX_DELAY));

    size_t samples = bytes_read / sizeof(int32_t);
    if (samples != FRAME_STEP)
{
    printf("Warning: expected %d samples, got %d\n",
           FRAME_STEP,
           (int)samples);
}

    for (size_t i = 0; i < samples; i++)
{
    pcm[i] = (int16_t)(i2s_buffer[i] >> 16);
}

printf("RAW: %ld %ld %ld %ld\n",
       (long)i2s_buffer[0],
       (long)i2s_buffer[1],
       (long)i2s_buffer[2],
       (long)i2s_buffer[3]);
    static int count = 0;

if (count++ % 5 == 0)
{
    printf("PCM: %d %d %d %d %d\n",
           pcm[0],
           pcm[1],
           pcm[2],
           pcm[3],
           pcm[4]);
}
}

static void tflite_init(void)
{
    model = tflite::GetModel(model_emergency_voice_int8_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("Model schema mismatch!\n");
        return;
    }

    static tflite::MicroMutableOpResolver<11> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();
    resolver.AddMul();
    resolver.AddAdd(); 
    resolver.AddMaxPool2D(); 
    resolver.AddQuantize();
    resolver.AddDequantize();
    static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize);

interpreter = &static_interpreter;

if (interpreter->AllocateTensors() != kTfLiteOk)
{
    printf("Tensor allocation failed!\n");
    return;
}

input = interpreter->input(0);
output = interpreter->output(0);

printf("\nInput scale      : %f\n", input->params.scale);
printf("Input zero point : %ld\n", static_cast<long>(input->params.zero_point));
printf("Output scale      : %f\n", output->params.scale);
printf("Output zero point : %ld\n", static_cast<long>(output->params.zero_point));
printf("\n========== MODEL INFO ==========\n");

printf("Input shape : ");
for (int i = 0; i < input->dims->size; i++)
{
    printf("%d ", input->dims->data[i]);
}

printf("\nInput bytes : %d", input->bytes);
printf("\nInput type  : %d\n", input->type);

printf("\nOutput shape : ");
for (int i = 0; i < output->dims->size; i++)
{
    printf("%d ", output->dims->data[i]);
}

printf("\nOutput bytes : %d", output->bytes);
printf("\nOutput type  : %d\n", output->type);

printf("===============================\n");

printf("TFLite initialized successfully!\n");
}

extern "C" void app_main(void)
{
    int16_t pcm[FRAME_STEP] = {0};

    frontend_init();
i2s_init();
tflite_init();

static int frames_collected = 0;

while (1)
{
    // Read microphone
    read_microphone(pcm);

    // Generate 49x64 Mel spectrogram
    frontend_process(pcm);

    

if (frames_collected < 49)
{
    frames_collected++;
    printf("Collecting spectrogram: %d/49\n", frames_collected);

    vTaskDelay(pdMS_TO_TICKS(20));
    continue;
}

    // Get pointer to spectrogram
    const float *mel = frontend_get_mel();

    // Copy 49x64 = 3136 values into input tensor
    for (int i = 0; i < 49 * 64; i++)
{
    float x = mel[i];

    int q = (int)roundf(x / input->params.scale)
            + input->params.zero_point;

    if (q > 127)
        q = 127;

    if (q < -128)
        q = -128;

    input->data.int8[i] = (int8_t)q;
}

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        printf("Inference failed!\n");
    }
    else
    {
        printf("Prediction: ");

        for (int i = 0; i < 35; i++)
        {
            printf("%d ", output->data.int8[i]);
        }

        printf("\n");
    }

    vTaskDelay(pdMS_TO_TICKS(20));
}
}