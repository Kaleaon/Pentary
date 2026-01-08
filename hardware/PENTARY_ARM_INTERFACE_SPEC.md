# Pentary-ARM Interface Specification

## Document Overview

This specification defines the communication interface between Pentary analog compute chips and recycled ARM processor chips in the hybrid blade architecture. This enables high-efficiency neural network inference using Pentary computing with ARM chips handling control, I/O, and coordination.

## 1. System Architecture

### 1.1 Hybrid Blade Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HYBRID BLADE SYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────┐         ┌───────────────────────────────────┐    │
│  │  ARM Control  │◄───────►│        COMMUNICATION BUS          │    │
│  │    Chip       │         │  (SPI/I2C/Custom Serial)           │    │
│  │               │         └───────────────────────────────────┘    │
│  │  - Cortex-A53 │                ▲   ▲   ▲   ▲                     │
│  │  - 4-8 cores  │                │   │   │   │                     │
│  │  - 1.2 GHz    │         ┌──────┘   │   │   └──────┐              │
│  │               │         │          │   │          │              │
│  └───────────────┘         ▼          ▼   ▼          ▼              │
│                      ┌─────────┐ ┌─────────┐    ┌─────────┐         │
│                      │Pentary  │ │Pentary  │... │Pentary  │         │
│                      │Chip 0   │ │Chip 1   │    │Chip N   │         │
│                      │         │ │         │    │         │         │
│                      │ 256 PEs │ │ 256 PEs │    │ 256 PEs │         │
│                      └─────────┘ └─────────┘    └─────────┘         │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    SHARED MEMORY BUS                           │  │
│  │              (DDR3/DDR4 from recycled RAM)                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Roles

| Component | Role | Interface Type |
|-----------|------|----------------|
| ARM Chip | System control, scheduling, I/O | Master |
| Pentary Chips | Neural computation (MAC ops) | Slave |
| System RAM | Data buffers, weights storage | Shared |
| Flash | Firmware, model storage | ARM-only |

## 2. Communication Protocol

### 2.1 Physical Layer

#### 2.1.1 Primary Interface: SPI (Serial Peripheral Interface)

```
ARM (Master)              Pentary (Slave)
     │                         │
     ├── SCLK ────────────────►│  Clock: 50 MHz max
     │                         │
     ├── MOSI ────────────────►│  Master Out Slave In
     │                         │
     │◄─────────────────MISO──┤  Master In Slave Out
     │                         │
     ├── CS_n ────────────────►│  Chip Select (active low)
     │                         │
     ├── INT ◄────────────────┤  Interrupt (computation done)
     │                         │
```

**SPI Configuration:**
- Mode: 0 (CPOL=0, CPHA=0)
- Clock frequency: Up to 50 MHz
- Data order: MSB first
- Word size: 8 bits

#### 2.1.2 Secondary Interface: I2C (for configuration)

- Clock: 400 kHz (Fast Mode)
- Used for: Initial configuration, status polling
- Address range: 0x20-0x2F (supports 16 Pentary chips)

### 2.2 Data Link Layer

#### 2.2.1 Packet Format

```
┌────────────┬────────────┬────────────┬────────────────┬──────────┐
│   Header   │  Command   │   Length   │    Payload     │   CRC    │
│  (1 byte)  │  (1 byte)  │  (2 bytes) │ (0-4096 bytes) │ (2 bytes)│
└────────────┴────────────┴────────────┴────────────────┴──────────┘
```

**Header Byte:**
```
Bit 7:   Direction (0=Write to Pentary, 1=Read from Pentary)
Bit 6:   Priority (0=Normal, 1=High)
Bit 5-4: Packet type (00=Data, 01=Command, 10=Status, 11=Config)
Bit 3-0: Sequence number
```

**Command Codes:**

| Code | Command | Description |
|------|---------|-------------|
| 0x01 | RESET | Reset Pentary chip |
| 0x02 | LOAD_WEIGHTS | Load neural network weights |
| 0x03 | COMPUTE | Start computation |
| 0x04 | READ_RESULT | Read computation results |
| 0x05 | SET_MODE | Configure operation mode |
| 0x06 | GET_STATUS | Read chip status |
| 0x07 | SET_POWER | Power management |
| 0x10 | BATCH_LOAD | Batch weight loading |
| 0x11 | BATCH_COMPUTE | Batch computation |
| 0x20 | DEBUG_READ | Debug register read |
| 0x21 | DEBUG_WRITE | Debug register write |

### 2.3 Data Encoding

#### 2.3.1 Pentary Data Format

Pentary values are encoded as 3-bit values packed into bytes:

```
Encoding:      Binary    Decimal  Pentary Digit
  -2            000         0         ⊖
  -1            001         1         -
   0            010         2         0
  +1            011         3         +
  +2            100         4         ⊕

Byte packing (2 digits per byte):
┌────────────────┬────────────────┐
│  Digit 0 (3b)  │  Digit 1 (3b)  │  Unused (2b)
└────────────────┴────────────────┘
```

#### 2.3.2 Weight Matrix Format

```c
struct weight_packet {
    uint8_t  row_start;      // Starting row in PE array
    uint8_t  col_start;      // Starting column
    uint8_t  rows;           // Number of rows
    uint8_t  cols;           // Number of columns
    uint8_t  data[];         // Packed pentary weights
};
```

#### 2.3.3 Activation Vector Format

```c
struct activation_packet {
    uint16_t length;         // Vector length
    uint8_t  precision;      // 0=4-digit, 1=8-digit, 2=16-digit
    uint8_t  reserved;
    uint8_t  data[];         // Packed pentary values
};
```

## 3. Memory Map

### 3.1 Pentary Chip Register Map

| Address | Name | R/W | Description |
|---------|------|-----|-------------|
| 0x00 | STATUS | R | Chip status register |
| 0x01 | CONTROL | R/W | Control register |
| 0x02 | CONFIG | R/W | Configuration register |
| 0x03 | INT_EN | R/W | Interrupt enable |
| 0x04 | INT_FLAG | R/C | Interrupt flags (read-clear) |
| 0x10-0x1F | PE_STATUS[16] | R | Per-PE status |
| 0x20-0x2F | PE_CTRL[16] | R/W | Per-PE control |
| 0x40-0x7F | WEIGHT_BASE | W | Weight memory base |
| 0x80-0xBF | ACT_BASE | R/W | Activation memory |
| 0xC0-0xFF | RESULT_BASE | R | Result memory |

### 3.2 Status Register (0x00)

```
Bit 7:   BUSY - Computation in progress
Bit 6:   DONE - Computation complete
Bit 5:   ERR - Error occurred
Bit 4:   OVF - Overflow detected
Bit 3-2: POWER_STATE (00=Active, 01=Idle, 10=Sleep, 11=Off)
Bit 1:   WEIGHTS_VALID - Weights loaded
Bit 0:   READY - Chip initialized and ready
```

### 3.3 Shared Memory Layout

```
┌─────────────────────────────────────────────────┐  0x00000000
│               ARM Code/Data                      │
│                 (128 MB)                         │
├─────────────────────────────────────────────────┤  0x08000000
│            Weight Buffer Pool                    │
│                 (256 MB)                         │
│  - Pre-quantized pentary weights                 │
│  - Supports 4 concurrent model layers            │
├─────────────────────────────────────────────────┤  0x18000000
│           Activation Buffer Pool                 │
│                 (128 MB)                         │
│  - Input activations                             │
│  - Intermediate results                          │
├─────────────────────────────────────────────────┤  0x20000000
│            Result Buffer Pool                    │
│                 (128 MB)                         │
│  - Pentary computation outputs                   │
│  - DMA transfer staging                          │
├─────────────────────────────────────────────────┤  0x28000000
│           System Reserved                        │
│                 (128 MB)                         │
└─────────────────────────────────────────────────┘  0x30000000
```

## 4. Operation Sequences

### 4.1 Initialization Sequence

```
ARM                              Pentary
 │                                  │
 │──── RESET ──────────────────────►│
 │                                  │  [Initialize hardware]
 │◄─── STATUS (READY=1) ───────────│
 │                                  │
 │──── SET_MODE (inference) ───────►│
 │                                  │
 │──── LOAD_WEIGHTS (batch 0) ─────►│
 │         [256 KB weights]         │
 │                                  │  [Store in local SRAM]
 │◄─── STATUS (WEIGHTS_VALID=1) ───│
 │                                  │
```

### 4.2 Inference Operation

```
ARM                              Pentary
 │                                  │
 │──── COMPUTE (layer params) ─────►│
 │         [activation data]        │
 │                                  │  [MAC operations]
 │                                  │  [Apply activation]
 │◄─── INT (DONE) ─────────────────│
 │                                  │
 │──── READ_RESULT ────────────────►│
 │                                  │
 │◄─── [result data] ──────────────│
 │                                  │
```

### 4.3 Pipelined Operation

For high throughput, use double-buffering:

```
Time    ARM Activity              Pentary Activity
────    ────────────              ────────────────
t0      Load weights batch 1      [Idle]
t1      Load activations 1        Store weights
t2      Start compute 1           [Configure]
t3      Load activations 2        Compute batch 1
t4      Start compute 2           Compute batch 1 (cont)
t5      Read results 1            Compute batch 2
t6      Process results 1         Compute batch 2 (cont)
t7      Load activations 3        [Return results 2]
...
```

## 5. Error Handling

### 5.1 Error Codes

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| 0x00 | OK | No error | N/A |
| 0x01 | CRC_ERR | CRC mismatch | Retransmit |
| 0x02 | TIMEOUT | Operation timeout | Reset |
| 0x03 | OVERFLOW | Arithmetic overflow | Check inputs |
| 0x04 | UNDERFLOW | Arithmetic underflow | Check inputs |
| 0x05 | INVALID_CMD | Unknown command | Check protocol |
| 0x06 | WEIGHT_ERR | Weight loading failed | Reload |
| 0x07 | MEM_ERR | Memory access error | Reset |
| 0x10 | PE_FAULT | Processing element fault | Remap |

### 5.2 Recovery Procedures

**Soft Recovery:**
1. Send GET_STATUS to identify error
2. Clear error flags with INT_FLAG write
3. Retry operation

**Hard Recovery:**
1. Assert RESET command
2. Wait 100ms
3. Re-run initialization sequence
4. Reload weights

## 6. Performance Specifications

### 6.1 Timing Requirements

| Operation | Typical | Maximum |
|-----------|---------|---------|
| SPI transaction setup | 20 ns | 50 ns |
| Weight load (256 KB) | 5 ms | 10 ms |
| Activation load (1 KB) | 20 µs | 50 µs |
| MAC operation (256 PEs) | 10 ns | 20 ns |
| Full layer (1M MACs) | 4 ms | 8 ms |
| Result read (1 KB) | 20 µs | 50 µs |

### 6.2 Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Peak MAC/s | 256 GMAC/s | All PEs active |
| Sustained MAC/s | 200 GMAC/s | Including I/O overhead |
| SPI bandwidth | 50 MB/s | 50 MHz clock |
| Power efficiency | 10 TOPS/W | At 1.8V operation |

## 7. Power Management

### 7.1 Power States

| State | Pentary Power | SPI Active | Wake Time |
|-------|---------------|------------|-----------|
| Active | Full | Yes | N/A |
| Idle | 50% | Yes | Immediate |
| Sleep | 10% | No | 100 µs |
| Off | 0% | No | 10 ms |

### 7.2 Power Control Commands

```c
// Set power state
void set_power_state(uint8_t chip_id, power_state_t state) {
    spi_write(chip_id, CMD_SET_POWER, &state, 1);
}

// Dynamic voltage scaling
void set_voltage(uint8_t chip_id, uint8_t mv_offset) {
    // mv_offset: 0-31 (0 = 1.8V, 31 = 1.5V)
    spi_write(chip_id, CMD_SET_VOLTAGE, &mv_offset, 1);
}
```

## 8. ARM Firmware API

### 8.1 Initialization

```c
// Initialize Pentary subsystem
int pentary_init(pentary_config_t* config);

// Configure individual chip
int pentary_chip_config(uint8_t chip_id, chip_config_t* config);

// Load pre-quantized weights
int pentary_load_weights(uint8_t chip_id, 
                         const pentary_weight_t* weights,
                         size_t size);
```

### 8.2 Inference

```c
// Single inference call
int pentary_inference(uint8_t chip_id,
                      const pentary_activation_t* input,
                      pentary_activation_t* output);

// Batched inference (pipelined)
int pentary_batch_inference(uint8_t* chip_ids,
                            size_t num_chips,
                            const pentary_batch_t* batch,
                            pentary_batch_t* results);
```

### 8.3 Status and Diagnostics

```c
// Get chip status
int pentary_get_status(uint8_t chip_id, pentary_status_t* status);

// Run self-test
int pentary_self_test(uint8_t chip_id, pentary_test_result_t* result);

// Read performance counters
int pentary_get_counters(uint8_t chip_id, pentary_counters_t* counters);
```

## 9. Reference Implementation

### 9.1 Weight Loading Example

```c
#include "pentary_api.h"

int load_layer_weights(uint8_t chip_id, nn_layer_t* layer) {
    // Convert float weights to pentary
    pentary_weight_t* pent_weights = quantize_to_pentary(
        layer->weights, 
        layer->weight_count
    );
    
    // Chunk into packets
    size_t offset = 0;
    while (offset < layer->weight_count) {
        size_t chunk = MIN(4096, layer->weight_count - offset);
        
        weight_packet_t pkt = {
            .row_start = offset / layer->cols,
            .col_start = offset % layer->cols,
            .rows = chunk / layer->cols,
            .cols = layer->cols
        };
        
        int ret = spi_write_packet(chip_id, CMD_LOAD_WEIGHTS,
                                   &pkt, sizeof(pkt),
                                   &pent_weights[offset], chunk);
        if (ret != 0) return ret;
        
        offset += chunk;
    }
    
    free(pent_weights);
    return 0;
}
```

### 9.2 Inference Pipeline Example

```c
int run_model_inference(nn_model_t* model, 
                        float* input, 
                        float* output) {
    pentary_activation_t act_buf[2];  // Double buffer
    int cur_buf = 0;
    
    // Quantize input
    quantize_activations(input, &act_buf[cur_buf], model->input_size);
    
    // Process each layer
    for (int layer = 0; layer < model->num_layers; layer++) {
        int next_buf = 1 - cur_buf;
        
        // Select chip for this layer (round-robin)
        uint8_t chip_id = layer % num_pentary_chips;
        
        // Run inference
        pentary_inference(chip_id, &act_buf[cur_buf], &act_buf[next_buf]);
        
        cur_buf = next_buf;
    }
    
    // Dequantize output
    dequantize_activations(&act_buf[cur_buf], output, model->output_size);
    
    return 0;
}
```

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-08 | Initial specification |

## Appendix A: Pin Assignments

| Pin | Function | I/O | Notes |
|-----|----------|-----|-------|
| 1 | VDD_CORE | PWR | 1.8V core |
| 2 | VDD_IO | PWR | 3.3V I/O |
| 3 | GND | PWR | Ground |
| 4 | SCLK | I | SPI clock |
| 5 | MOSI | I | SPI data in |
| 6 | MISO | O | SPI data out |
| 7 | CS_n | I | Chip select |
| 8 | INT | O | Interrupt output |
| 9-12 | A[3:0] | I | Address select |
| 13 | RESET_n | I | Active-low reset |
| 14-15 | Reserved | - | Future use |
| 16 | NC | - | No connect |

## Appendix B: CRC Calculation

Use CRC-16-CCITT (polynomial 0x1021):

```c
uint16_t crc16_ccitt(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
        }
    }
    return crc;
}
```
