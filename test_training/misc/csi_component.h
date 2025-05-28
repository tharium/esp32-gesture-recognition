/* Modified from original work by Steven M. Hernandez
 * Source: https://github.com/StevenMHernandez/ESP32-CSI-Tool
 * Licensed under the MIT License
 *
 * MIT License
 * Copyright (c) 2020 Steven M. Hernandez
 * MIT License

	Copyright (c) 2020 Steven M. Hernandez

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
 */

#ifndef ESP32_CSI_CSI_COMPONENT_H
#define ESP32_CSI_CSI_COMPONENT_H

#include "time_component.h"
#include "math.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <sstream>
#include <iostream>

// Config
#define CSI_RAW 1
#define CSI_AMPLITUDE 0
#define CSI_PHASE 0
#define CSI_TYPE CSI_RAW

// Queue config
#define CSI_QUEUE_SIZE 32
#define CSI_TASK_STACK_SIZE 4096
#define CSI_TASK_PRIORITY 5
#define CSI_TASK_CORE 1  // Pin to core 1

// Global
static char *project_type = NULL;
static QueueHandle_t csi_queue = NULL;
static TaskHandle_t csi_task_handle = NULL;
static bool csi_initialized = false;

// Struct to hold CSI data for queue
typedef struct {
    wifi_csi_info_t info;
    uint8_t csi_data[384];  // Max size
    uint32_t timestamp;
    bool real_time_set_snapshot;
} csi_queue_item_t;

// Fast interrupt callback
void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
    if (!csi_queue || !data) return;
    
    static csi_queue_item_t queue_item;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    // Copy data 
    queue_item.info = *data;
    queue_item.timestamp = get_steady_clock_timestamp();
    queue_item.real_time_set_snapshot = real_time_set;
    
    // Copy CSI data buffer
    size_t copy_len = data->len;
    if (copy_len > sizeof(queue_item.csi_data)) {
        copy_len = sizeof(queue_item.csi_data);
    }
    memcpy(queue_item.csi_data, data->buf, copy_len);
    
    // Send to queue (non-blocking)
    if (xQueueSendFromISR(csi_queue, &queue_item, &xHigherPriorityTaskWoken) != pdTRUE) {
        // Queue full - packet dropped
    }
    
    // Yield to higher priority task if woken
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// Processing task
void csi_processing_task(void *param) {
    csi_queue_item_t queue_item;
    char mac_str[20];
    
    // Use stringstream for string building
    std::stringstream ss;
    
    while (1) {
        // Wait for data
        if (xQueueReceive(csi_queue, &queue_item, portMAX_DELAY) == pdTRUE) {
            wifi_csi_info_t *d = &queue_item.info;
            
            // Clear and reuse stringstream
            ss.str("");
            ss.clear();
            
            // Format MAC address
            snprintf(mac_str, sizeof(mac_str), "%02X:%02X:%02X:%02X:%02X:%02X", 
                    d->mac[0], d->mac[1], d->mac[2], d->mac[3], d->mac[4], d->mac[5]);
            
            // Build CSV string
            ss << "CSI_DATA,"
               << project_type << ","
               << mac_str << ","
               << d->rx_ctrl.rssi << ","
               << d->rx_ctrl.rate << ","
               << d->rx_ctrl.sig_mode << ","
               << d->rx_ctrl.mcs << ","
               << d->rx_ctrl.cwb << ","
               << d->rx_ctrl.smoothing << ","
               << d->rx_ctrl.not_sounding << ","
               << d->rx_ctrl.aggregation << ","
               << d->rx_ctrl.stbc << ","
               << d->rx_ctrl.fec_coding << ","
               << d->rx_ctrl.sgi << ","
               << d->rx_ctrl.noise_floor << ","
               << d->rx_ctrl.ampdu_cnt << ","
               << d->rx_ctrl.channel << ","
               << d->rx_ctrl.secondary_channel << ","
               << d->rx_ctrl.timestamp << ","
               << d->rx_ctrl.ant << ","
               << d->rx_ctrl.sig_len << ","
               << d->rx_ctrl.rx_state << ","
               << queue_item.real_time_set_snapshot << ","
               << queue_item.timestamp << ","
               << d->len << ",[";

#if CONFIG_SHOULD_COLLECT_ONLY_LLTF
            int data_len = 128;
#else
            int data_len = d->len;
#endif

            // Process CSI data based on type
            int8_t *csi_ptr = (int8_t*)queue_item.csi_data;
            
#if CSI_RAW
            for (int i = 0; i < data_len && i < sizeof(queue_item.csi_data); i++) {
                ss << (int)csi_ptr[i];
                if (i < data_len - 1) ss << " ";
            }
#endif

#if CSI_AMPLITUDE
            for (int i = 0; i < data_len / 2 && (i * 2 + 1) < sizeof(queue_item.csi_data); i++) {
                // Use integer math for better performance
                int real = csi_ptr[i * 2];
                int imag = csi_ptr[i * 2 + 1];
                int amplitude = (int)sqrt(real * real + imag * imag);
                ss << amplitude;
                if (i < (data_len / 2) - 1) ss << " ";
            }
#endif

#if CSI_PHASE
            for (int i = 0; i < data_len / 2 && (i * 2 + 1) < sizeof(queue_item.csi_data); i++) {
                int phase = (int)(atan2(csi_ptr[i * 2 + 1], csi_ptr[i * 2]) * 180.0 / M_PI);
                ss << phase;
                if (i < (data_len / 2) - 1) ss << " ";
            }
#endif

            ss << "]\n";
            
            // Output
            printf("%s", ss.str().c_str());
            fflush(stdout);
            
            // Delay
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

void print_csi_csv_header() {
    const char *header_str = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA\n";
    printf("%s", header_str);
    fflush(stdout);
}

void csi_init(char *type) {
    if (csi_initialized) {
        ESP_LOGW("CSI", "CSI already initialized");
        return;
    }
    
    project_type = type;
    
#ifdef CONFIG_SHOULD_COLLECT_CSI
    ESP_LOGI("CSI", "Initializing CSI collection...");
    
    // Queue for CSI data
    csi_queue = xQueueCreate(CSI_QUEUE_SIZE, sizeof(csi_queue_item_t));
    if (csi_queue == NULL) {
        ESP_LOGE("CSI", "Failed to create CSI queue");
        return;
    }
    
    // Create task
    BaseType_t task_result = xTaskCreatePinnedToCore(
        csi_processing_task,
        "csi_proc",
        CSI_TASK_STACK_SIZE,
        NULL,
        CSI_TASK_PRIORITY,
        &csi_task_handle,
        CSI_TASK_CORE
    );
    
    if (task_result != pdPASS) {
        ESP_LOGE("CSI", "Failed to create CSI processing task");
        vQueueDelete(csi_queue);
        return;
    }
    
    // Configure CSI
    ESP_ERROR_CHECK(esp_wifi_set_csi(1));
    
    wifi_csi_config_t configuration_csi = {
        .lltf_en = 1,
        .htltf_en = 1, 
        .stbc_htltf2_en = 1,
        .ltf_merge_en = 1,
        .channel_filter_en = 0,
        .manu_scale = 0,
        .shift = 0,
        .dump_ack_en = 0
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&configuration_csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_cb, NULL));
    
    print_csi_csv_header();
    csi_initialized = true;
    
    ESP_LOGI("CSI", "CSI collection initialized successfully");
#else
    ESP_LOGI("CSI", "CSI collection disabled in config");
#endif
}

void csi_deinit() {
    if (!csi_initialized) return;
    
#ifdef CONFIG_SHOULD_COLLECT_CSI
    esp_wifi_set_csi(0);
    esp_wifi_set_csi_rx_cb(NULL, NULL);
    
    if (csi_task_handle) {
        vTaskDelete(csi_task_handle);
        csi_task_handle = NULL;
    }
    
    if (csi_queue) {
        vQueueDelete(csi_queue);
        csi_queue = NULL;
    }
    
    csi_initialized = false;
    ESP_LOGI("CSI", "CSI collection deinitialized");
#endif
}

#endif //ESP32_CSI_CSI_COMPONENT_H
