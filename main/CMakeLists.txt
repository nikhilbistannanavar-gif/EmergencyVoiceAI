idf_component_register(
    SRCS
        "EmergencyVoiceAI_V3.cpp"
        "../model/model_data.cc"

    INCLUDE_DIRS
        "."
        "../model"

    REQUIRES
        frontend
        esp_psram
        driver
        espressif__esp-tflite-micro
)