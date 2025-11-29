# Pentary Memory Model Specification

## 1. Overview

This document defines the memory model for the Pentary architecture, including address spaces, memory types, coherency protocols, and atomic operations.

### 1.1 Address Space

| Address Width | Physical Bits | Addressable Space | Use Case |
|---------------|---------------|-------------------|----------|
| 16 pents | 48 bits | 281 TB | Standard mode |
| 20 pents | 60 bits | 1.15 EB | Extended mode |
| 28 pents | 84 bits | 19.8 YB | Full 64-bit equivalent |

### 1.2 Default Configuration

- **Virtual Address**: 20 pents (≈46 bits, 64 TB addressable)
- **Physical Address**: 16 pents (≈37 bits, 137 GB addressable per chip)
- **Page Size**: 4096 pents (≈12 KB)
- **Huge Page**: 5^8 pents (≈390,625 KB ≈ 381 MB)

## 2. Memory Hierarchy

### 2.1 Complete Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Memory Hierarchy                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    Register File                                  │          │
│  │  32 × P16 GPRs + 32 × V256 Vector Regs                           │          │
│  │  Access Time: 0 cycles                                            │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    L1 Data Cache                                  │          │
│  │  Size: 32 KB per core                                             │          │
│  │  Line Size: 64 pents (192 bytes)                                  │          │
│  │  Associativity: 4-way set associative                             │          │
│  │  Access Time: 1-2 cycles                                          │          │
│  │  Policy: Write-through                                            │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    L1 Instruction Cache                           │          │
│  │  Size: 32 KB per core                                             │          │
│  │  Line Size: 64 pents                                              │          │
│  │  Associativity: 4-way                                             │          │
│  │  Access Time: 1 cycle                                             │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    L2 Cache (Unified)                             │          │
│  │  Size: 256 KB per core                                            │          │
│  │  Line Size: 128 pents (384 bytes)                                 │          │
│  │  Associativity: 8-way                                             │          │
│  │  Access Time: 8-12 cycles                                         │          │
│  │  Policy: Write-back                                               │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    L3 Cache (Shared)                              │          │
│  │  Size: 8 MB total (shared by all cores)                           │          │
│  │  Line Size: 256 pents (768 bytes)                                 │          │
│  │  Associativity: 16-way                                            │          │
│  │  Access Time: 30-40 cycles                                        │          │
│  │  Policy: Write-back, inclusive                                    │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    Main Memory (DRAM/Memristor)                   │          │
│  │  Access Time: 100-200 cycles                                      │          │
│  │  Bandwidth: 100 GB/s per channel                                  │          │
│  │  Channels: 4-8                                                    │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                              ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    Storage (NVMe/SSD)                             │          │
│  │  Access Time: 10,000+ cycles                                      │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Memory Regions

### 3.1 Virtual Address Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    20-Pent Virtual Address Space                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  0x00000_00000_00000_00000  ┌────────────────────────┐                          │
│                             │    NULL Guard Page     │  4 KB                    │
│  0x00000_00000_00000_01000  ├────────────────────────┤                          │
│                             │                        │                          │
│                             │    User Code (.text)   │  Variable                │
│                             │                        │                          │
│                             ├────────────────────────┤                          │
│                             │    Read-Only Data      │                          │
│                             │    (.rodata)           │  Variable                │
│                             ├────────────────────────┤                          │
│                             │    Read-Write Data     │                          │
│                             │    (.data, .bss)       │  Variable                │
│                             ├────────────────────────┤                          │
│                             │         Heap           │                          │
│                             │         ↓              │  Grows down              │
│                             │                        │                          │
│                             │    (Free space)        │                          │
│                             │                        │                          │
│                             │         ↑              │  Grows up                │
│                             │        Stack           │                          │
│  0x7FFFF_FFFFF_FFFFF_FF000  ├────────────────────────┤                          │
│                             │    Stack Guard Page    │  4 KB                    │
│  0x80000_00000_00000_00000  ├────────────────────────┤                          │
│                             │                        │                          │
│                             │    Kernel Space        │  Half of address space   │
│                             │                        │                          │
│  0xFFFFF_FFFFF_FFFFF_FFFFF  └────────────────────────┘                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Types

| Type | Caching | Ordering | Use Case |
|------|---------|----------|----------|
| Normal | Write-back | Relaxed | General data |
| Device | Uncached | Strict | Memory-mapped I/O |
| Neural | Compute-in-place | Relaxed | Weight matrices |
| Stack | Write-through | Strict | Stack memory |

## 4. Virtual Memory

### 4.1 Page Table Structure

**3-Level Page Table** (for 20-pent virtual addresses):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Page Table Entry (16 pents)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Bits [0-11]:     Physical Frame Number (12 pents)                              │
│  Bits [12]:       Valid (V)                                                     │
│  Bits [13]:       Readable (R)                                                  │
│  Bits [14]:       Writable (W)                                                  │
│  Bits [15]:       Executable (X)                                                │
│  Bits [16]:       User accessible (U)                                           │
│  Bits [17]:       Global (G) - not flushed on context switch                    │
│  Bits [18]:       Accessed (A)                                                  │
│  Bits [19]:       Dirty (D)                                                     │
│  Bits [20-23]:    Reserved                                                      │
│  Bits [24-27]:    Memory Type                                                   │
│  Bits [28-31]:    Reserved for extensions                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Translation Process

```
Virtual Address (20 pents):
┌───────┬───────┬───────┬────────────┐
│ L3 Idx│ L2 Idx│ L1 Idx│   Offset   │
│5 pents│5 pents│5 pents│  5 pents   │
└───────┴───────┴───────┴────────────┘

Page Walk:
1. L3 Index → L3 Table Entry → L2 Table Base
2. L2 Index → L2 Table Entry → L1 Table Base
3. L1 Index → L1 Table Entry → Physical Frame
4. Offset → Byte within page
```

### 4.3 TLB (Translation Lookaside Buffer)

```
L1 TLB:
  - 64 entries (fully associative)
  - Separate I-TLB and D-TLB
  - 1-cycle hit latency

L2 TLB:
  - 1024 entries (4-way associative)
  - Unified instruction/data
  - 4-cycle hit latency
```

## 5. Cache Coherency

### 5.1 MESIF Protocol

Extended MESI with Forward state for pentary multiprocessor systems:

| State | Meaning | Valid | Exclusive | Modified | Forward |
|-------|---------|-------|-----------|----------|---------|
| M | Modified | Yes | Yes | Yes | No |
| E | Exclusive | Yes | Yes | No | No |
| S | Shared | Yes | No | No | No |
| I | Invalid | No | - | - | No |
| F | Forward | Yes | No | No | Yes |

### 5.2 Coherency Operations

```
; Cache control instructions
FLUSH   addr            ; Flush cache line to memory
CLEAN   addr            ; Write back if dirty, keep valid
INVAL   addr            ; Invalidate cache line
ZERO    addr            ; Allocate and zero cache line (no read)
```

### 5.3 Snoop Protocol

```
Bus Transactions:
- BusRd: Read request (transition to S)
- BusRdX: Read-exclusive request (transition to M)
- BusUpgr: Upgrade request (S→M without data)
- Flush: Write back to memory
```

## 6. Memory Ordering

### 6.1 Default Ordering (Relaxed)

Default pentary memory model is relaxed for performance:
- Loads can be reordered with other loads
- Stores can be reordered with other stores
- Loads can be reordered with prior stores (to different addresses)

### 6.2 Ordering Instructions

```
; Memory barriers
FENCE.R           ; Order all prior reads before subsequent reads
FENCE.W           ; Order all prior writes before subsequent writes
FENCE.RW          ; Order all prior reads before subsequent writes
FENCE.WR          ; Order all prior writes before subsequent reads
FENCE             ; Full memory barrier (order everything)

; Acquire/Release semantics
LOAD.ACQ  Rd, [addr]    ; Load with acquire semantics
STORE.REL [addr], Rs    ; Store with release semantics
```

### 6.3 Sequential Consistency (Optional)

```
; Enable sequential consistency mode (slower)
SETMODE SC              ; All subsequent memory ops are sequentially consistent
```

## 7. Atomic Operations

### 7.1 Load-Reserved / Store-Conditional

```
; Load-reserved (marks address for monitoring)
LR.P16  Rd, [addr]      ; Load and reserve P16
LR.P28  Dd, [addr]      ; Load and reserve P28

; Store-conditional (succeeds only if reservation held)
SC.P16  Rs, [addr], Rd  ; Store if reserved; Rd = 1 on success, 0 on fail
SC.P28  Ds, [addr], Rd  ; Store-conditional P28
```

### 7.2 Atomic Memory Operations (AMOs)

```
; Atomic fetch-and-operate
AMOADD.P16  Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] += Rs
AMOSUB.P16  Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] -= Rs
AMOAND.P16  Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] &= Rs (min)
AMOOR.P16   Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] |= Rs (max)
AMOSWAP.P16 Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] = Rs
AMOMIN.P16  Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] = min(Mem, Rs)
AMOMAX.P16  Rd, Rs, [addr]   ; Rd = Mem[addr]; Mem[addr] = max(Mem, Rs)

; Compare-and-swap
CAS.P16     Rd, Rs, Rt, [addr]  ; if (Mem == Rs) Mem = Rt; Rd = old Mem
```

## 8. Memory-Mapped I/O

### 8.1 I/O Address Space

```
Addresses 0xF0000_00000_00000_00000 - 0xFFFFF_FFFFF_FFFFF_FFFF:
  Reserved for memory-mapped I/O

I/O Regions:
  0xF0000... - 0xF0FFF...: Timer and interrupt controller
  0xF1000... - 0xF1FFF...: UART and serial ports
  0xF2000... - 0xF2FFF...: Network interfaces
  0xF3000... - 0xF3FFF...: Storage controllers
  0xF4000... - 0xF4FFF...: Neural accelerator control
  0xF5000... - 0xFFFF...: Reserved for expansion
```

### 8.2 I/O Instructions

```
; I/O with strict ordering
IO.RD.P16   Rd, [io_addr]   ; Read from I/O (non-cached, ordered)
IO.WR.P16   Rs, [io_addr]   ; Write to I/O (non-cached, ordered)

; DMA setup
DMA.SRC     Rd, addr        ; Set DMA source address
DMA.DST     Rd, addr        ; Set DMA destination address
DMA.LEN     Rd, len         ; Set DMA transfer length
DMA.START   mode            ; Start DMA transfer
DMA.WAIT                    ; Wait for DMA completion
```

## 9. In-Memory Computing

### 9.1 Memristor Crossbar Integration

```
Memory Map for Crossbar Arrays:
  Base + 0x0000: Crossbar 0 (256×256 weights)
  Base + 0x1000: Crossbar 1
  ...

Crossbar Control:
  CROSS.WRITE [addr], row, col, val   ; Write weight to crossbar
  CROSS.READ  Rd, [addr], row, col    ; Read weight from crossbar
  CROSS.MATVEC Vd, [addr], Vs         ; Matrix-vector multiply
```

### 9.2 Compute-in-Memory Operations

```
; Trigger in-memory computation
PIM.ADD     [addr], Rs1, Rs2    ; Mem[addr] += Rs1 + Rs2 (in-place)
PIM.MAC     [addr], Rs1, Rs2    ; Mem[addr] += Rs1 × Rs2 (in-place)
PIM.THRESH  [addr], thresh      ; Mem[addr] = (Mem[addr] > thresh) ? +2 : -2
```

## 10. Memory Protection

### 10.1 Page-Level Protection

Protection bits in page table entries:
- R (Read): Page is readable
- W (Write): Page is writable
- X (Execute): Page contains executable code
- U (User): Page accessible in user mode

### 10.2 Protection Keys

```
; Memory protection keys (4-bit key per page)
WRPKR   Rd, key, perms      ; Set permissions for key
RDPKR   Rd, key             ; Read permissions for key

; Permissions: RWX bits for each key
```

### 10.3 Memory Encryption (Optional)

```
; Encrypted memory regions
ENCRYPT.REGION  base, size, key   ; Mark region as encrypted
DECRYPT.REGION  base, size        ; Decrypt and remove encryption

; Transparent memory encryption (TME)
; Hardware encrypts/decrypts on the fly
```

## 11. Memory Allocation Hints

### 11.1 Allocation Instructions

```
; Allocation hints to OS/allocator
ALLOC.HINT  Rd, size, type   ; Request allocation with hints
; type: NORMAL, HUGE_PAGE, PINNED, SHARED, NEURAL

; Deallocation
FREE.HINT   addr             ; Mark memory as reclaimable
```

### 11.2 Prefetch Instructions

```
; Software prefetch
PREFETCH.R  [addr]          ; Prefetch for read
PREFETCH.W  [addr]          ; Prefetch for write
PREFETCH.NN [addr]          ; Prefetch to neural memory
```

---

**Document Version**: 1.0
**Status**: Specification Complete
