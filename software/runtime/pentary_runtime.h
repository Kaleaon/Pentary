/**
 * @file pentary_runtime.h
 * @brief Pentary Runtime Library API
 * 
 * This library provides low-level access to the Pentary hardware accelerator,
 * including memory management, kernel dispatch, and synchronization primitives.
 * 
 * @version 1.0
 * @date 2026-01-03
 */

#ifndef PENTARY_RUNTIME_H
#define PENTARY_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types and Constants
// ============================================================================

typedef enum {
    PENTARY_SUCCESS = 0,
    PENTARY_ERROR_INVALID_DEVICE = -1,
    PENTARY_ERROR_OUT_OF_MEMORY = -2,
    PENTARY_ERROR_INVALID_VALUE = -3,
    PENTARY_ERROR_NOT_INITIALIZED = -4,
    PENTARY_ERROR_KERNEL_FAILED = -5,
} pentary_status_t;

typedef struct pentary_device* pentary_device_t;
typedef struct pentary_stream* pentary_stream_t;
typedef struct pentary_event* pentary_event_t;
typedef void* pentary_ptr_t;

typedef enum {
    PENTARY_MEM_HBM,      // High Bandwidth Memory (HBM3)
    PENTARY_MEM_ONCHIP,   // On-chip 3T memory
    PENTARY_MEM_HOST,     // Host (CPU) memory
} pentary_mem_type_t;

// ============================================================================
// Device Management
// ============================================================================

/**
 * @brief Get the number of available Pentary devices
 * @param count Output: number of devices
 * @return Status code
 */
pentary_status_t pentary_get_device_count(int* count);

/**
 * @brief Initialize a Pentary device
 * @param device_id Device ID (0-based)
 * @param device Output: device handle
 * @return Status code
 */
pentary_status_t pentary_device_init(int device_id, pentary_device_t* device);

/**
 * @brief Destroy a Pentary device handle
 * @param device Device handle
 * @return Status code
 */
pentary_status_t pentary_device_destroy(pentary_device_t device);

/**
 * @brief Synchronize all operations on a device
 * @param device Device handle
 * @return Status code
 */
pentary_status_t pentary_device_synchronize(pentary_device_t device);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * @brief Allocate memory on the Pentary device
 * @param device Device handle
 * @param size Size in bytes
 * @param type Memory type (HBM, on-chip, etc.)
 * @param ptr Output: pointer to allocated memory
 * @return Status code
 */
pentary_status_t pentary_malloc(pentary_device_t device, size_t size,
                                 pentary_mem_type_t type, pentary_ptr_t* ptr);

/**
 * @brief Free memory on the Pentary device
 * @param ptr Pointer to memory
 * @return Status code
 */
pentary_status_t pentary_free(pentary_ptr_t ptr);

/**
 * @brief Copy memory from host to device
 * @param dst Destination (device) pointer
 * @param src Source (host) pointer
 * @param size Size in bytes
 * @param stream Stream for asynchronous operation (NULL for synchronous)
 * @return Status code
 */
pentary_status_t pentary_memcpy_h2d(pentary_ptr_t dst, const void* src,
                                     size_t size, pentary_stream_t stream);

/**
 * @brief Copy memory from device to host
 * @param dst Destination (host) pointer
 * @param src Source (device) pointer
 * @param size Size in bytes
 * @param stream Stream for asynchronous operation (NULL for synchronous)
 * @return Status code
 */
pentary_status_t pentary_memcpy_d2h(void* dst, pentary_ptr_t src,
                                     size_t size, pentary_stream_t stream);

/**
 * @brief Copy memory from device to device
 * @param dst Destination (device) pointer
 * @param src Source (device) pointer
 * @param size Size in bytes
 * @param stream Stream for asynchronous operation (NULL for synchronous)
 * @return Status code
 */
pentary_status_t pentary_memcpy_d2d(pentary_ptr_t dst, pentary_ptr_t src,
                                     size_t size, pentary_stream_t stream);

// ============================================================================
// Stream Management
// ============================================================================

/**
 * @brief Create a stream for asynchronous operations
 * @param device Device handle
 * @param stream Output: stream handle
 * @return Status code
 */
pentary_status_t pentary_stream_create(pentary_device_t device,
                                        pentary_stream_t* stream);

/**
 * @brief Destroy a stream
 * @param stream Stream handle
 * @return Status code
 */
pentary_status_t pentary_stream_destroy(pentary_stream_t stream);

/**
 * @brief Synchronize a stream (wait for all operations to complete)
 * @param stream Stream handle
 * @return Status code
 */
pentary_status_t pentary_stream_synchronize(pentary_stream_t stream);

// ============================================================================
// Event Management
// ============================================================================

/**
 * @brief Create an event for synchronization
 * @param device Device handle
 * @param event Output: event handle
 * @return Status code
 */
pentary_status_t pentary_event_create(pentary_device_t device,
                                       pentary_event_t* event);

/**
 * @brief Destroy an event
 * @param event Event handle
 * @return Status code
 */
pentary_status_t pentary_event_destroy(pentary_event_t event);

/**
 * @brief Record an event in a stream
 * @param event Event handle
 * @param stream Stream handle
 * @return Status code
 */
pentary_status_t pentary_event_record(pentary_event_t event,
                                       pentary_stream_t stream);

/**
 * @brief Wait for an event to complete
 * @param event Event handle
 * @return Status code
 */
pentary_status_t pentary_event_synchronize(pentary_event_t event);

/**
 * @brief Get elapsed time between two events (in milliseconds)
 * @param start Start event
 * @param end End event
 * @param elapsed_ms Output: elapsed time in milliseconds
 * @return Status code
 */
pentary_status_t pentary_event_elapsed_time(pentary_event_t start,
                                             pentary_event_t end,
                                             float* elapsed_ms);

// ============================================================================
// Kernel Dispatch
// ============================================================================

/**
 * @brief Launch a custom kernel on the Pentary cores
 * @param device Device handle
 * @param kernel_func Pointer to kernel function
 * @param args Kernel arguments
 * @param stream Stream for asynchronous execution
 * @return Status code
 */
pentary_status_t pentary_launch_kernel(pentary_device_t device,
                                        void (*kernel_func)(void*),
                                        void* args,
                                        pentary_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif // PENTARY_RUNTIME_H
