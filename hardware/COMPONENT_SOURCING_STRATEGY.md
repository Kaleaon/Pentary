# Component Sourcing and Testing Strategy

## Document Overview

This document outlines strategies for sourcing, testing, and qualifying recycled electronic components for use in Pentary hybrid blade systems. The goal is to reduce costs and environmental impact while maintaining system reliability.

## 1. Component Categories

### 1.1 Target Components

| Category | Source Devices | Target Specs | Quality Grade |
|----------|---------------|--------------|---------------|
| ARM Processors | Smartphones, tablets | Cortex-A53/A55 1.2+ GHz | A/B/C |
| LPDDR RAM | Smartphones | LPDDR4 2-8 GB | A/B |
| DDR RAM | Desktop PCs, laptops | DDR4 4-16 GB | A/B |
| Power Management | Various mobile | PMICs, regulators | A/B |
| Flash Storage | Various | eMMC, SD cards | A/B |

### 1.2 Component Grading System

**Grade A: Premium**
- Full specification performance
- < 1% defect rate in testing
- Suitable for production systems
- Premium pricing

**Grade B: Standard**
- 90-100% specification performance
- < 5% defect rate in testing
- Suitable for development, non-critical use
- Standard pricing

**Grade C: Economy**
- 80-95% specification performance
- < 10% defect rate in testing
- Suitable for testing, prototypes
- Budget pricing

**Grade F: Failed**
- Does not meet minimum specifications
- Recycled for materials

## 2. Sourcing Channels

### 2.1 Primary Sources

#### 2.1.1 E-Waste Recycling Centers

**Advantages:**
- High volume availability
- Low acquisition cost
- Established logistics

**Challenges:**
- Variable quality
- Mixed component ages
- Sorting required

**Key Partners to Develop:**
- Municipal e-waste programs
- Certified e-waste recyclers (R2/e-Stewards certified)
- Electronic waste aggregators

#### 2.1.2 Phone Repair Shops

**Advantages:**
- Known device history
- Pre-sorted by model
- Higher quality average

**Challenges:**
- Lower volume
- Higher per-unit cost
- Geographic distribution

**Approach:**
- Partner with repair shop networks
- Focus on "unrepairable" devices
- Offer bulk purchase agreements

#### 2.1.3 Corporate IT Disposal

**Advantages:**
- Consistent quality
- Bulk quantities
- Documentation available

**Challenges:**
- Scheduled availability
- Data security concerns
- Competitive bidding

**Approach:**
- ITAD (IT Asset Disposition) partnerships
- Offer secure destruction certificates
- Focus on 3-5 year old equipment

### 2.2 Secondary Sources

#### 2.2.1 Online Marketplaces

- eBay (bulk lots)
- Alibaba (wholesale)
- Specialized e-waste brokers

#### 2.2.2 Auction Houses

- Government surplus auctions
- Corporate liquidation sales
- University equipment disposal

### 2.3 Geographic Considerations

| Region | Advantages | Challenges |
|--------|------------|------------|
| North America | High-end devices, regulatory clarity | Higher labor costs |
| Europe | Strong recycling infrastructure | WEEE compliance |
| Asia | Low cost, high volume | Shipping, quality control |

## 3. Extraction Procedures

### 3.1 Smartphone Disassembly

#### 3.1.1 Equipment Required

- Hot air rework station (200-400°C)
- Precision screwdriver set
- Plastic pry tools (to avoid damage)
- ESD protection (mat, wrist strap)
- Fume extraction system
- Microscope or magnifier

#### 3.1.2 Disassembly Steps

```
1. Battery Removal
   - Discharge to <20% before handling
   - Use plastic tools to avoid puncture
   - Store in fireproof container

2. Display Separation
   - Heat edges to 80-100°C
   - Slowly pry adhesive
   - Set aside for separate recycling

3. Board Extraction
   - Remove all screws (document locations)
   - Disconnect flex cables carefully
   - Extract main PCB

4. Component Desoldering
   - SoC: Heat to 280°C, lift with vacuum
   - RAM: Lower temp (260°C) to avoid damage
   - PMICs: Same as RAM
   - Use flux and proper profiles
```

#### 3.1.3 Temperature Profiles

**BGA Package Desoldering:**
```
Phase 1: Preheat     25°C → 150°C   @ 1.5°C/s   (90 sec)
Phase 2: Soak        150°C → 200°C  @ 0.5°C/s   (60 sec)
Phase 3: Reflow      200°C → 280°C  @ 2.0°C/s   (40 sec)
Phase 4: Peak        280°C hold                  (10 sec)
Phase 5: Cooling     280°C → 100°C  @ 3.0°C/s   (60 sec)
```

### 3.2 Desktop RAM Extraction

**Much simpler process:**
1. Power off system, unplug
2. Open case
3. Release DIMM retention clips
4. Pull module straight up
5. Store in anti-static bag

### 3.3 Component Cleaning

**Post-extraction cleaning:**
1. Flux removal with isopropyl alcohol (99%)
2. Ultrasonic bath (5 minutes)
3. Air dry completely
4. Visual inspection under magnification
5. Store in anti-static packaging with desiccant

## 4. Testing Procedures

### 4.1 ARM Processor Testing

#### 4.1.1 Test Fixture Requirements

- Socket adapter for target package (BGA)
- Minimum viable board (power, clocks, JTAG)
- JTAG debugger (J-Link, DAPLINK)
- Power supply with current monitoring
- Temperature chamber (optional for stress test)

#### 4.1.2 Test Sequence

```python
class ARMProcessorTest:
    """Test sequence for recycled ARM processors"""
    
    def __init__(self, jtag_interface):
        self.jtag = jtag_interface
        self.results = {}
    
    def run_all_tests(self):
        """Execute complete test suite"""
        tests = [
            self.test_power_on,
            self.test_jtag_connection,
            self.test_cpu_id,
            self.test_core_functionality,
            self.test_memory_controller,
            self.test_thermal_behavior,
            self.test_clock_speeds,
        ]
        
        for test in tests:
            try:
                result = test()
                self.results[test.__name__] = result
            except Exception as e:
                self.results[test.__name__] = f"FAIL: {e}"
        
        return self.calculate_grade()
    
    def test_power_on(self):
        """Verify power consumption is within spec"""
        # Measure idle current
        idle_ma = self.jtag.measure_current()
        if idle_ma < 10 or idle_ma > 500:
            return "FAIL: Idle current out of range"
        return f"PASS: {idle_ma} mA"
    
    def test_jtag_connection(self):
        """Verify JTAG communication"""
        if not self.jtag.connect():
            return "FAIL: No JTAG response"
        return "PASS: JTAG connected"
    
    def test_cpu_id(self):
        """Read and verify CPU identification"""
        cpu_id = self.jtag.read_cpuid()
        if cpu_id == 0 or cpu_id == 0xFFFFFFFF:
            return "FAIL: Invalid CPU ID"
        return f"PASS: CPU ID = {hex(cpu_id)}"
    
    def test_core_functionality(self):
        """Run basic instruction tests on each core"""
        cores = self.jtag.get_core_count()
        failed_cores = []
        
        for core in range(cores):
            # Load and run simple test program
            self.jtag.select_core(core)
            if not self.run_core_test(core):
                failed_cores.append(core)
        
        if failed_cores:
            return f"FAIL: Cores {failed_cores} non-functional"
        return f"PASS: All {cores} cores functional"
    
    def test_memory_controller(self):
        """Test memory interface"""
        # Write pattern to SRAM
        pattern = [0xDEADBEEF, 0xCAFEBABE, 0x12345678]
        for i, val in enumerate(pattern):
            self.jtag.write_mem(0x20000000 + i*4, val)
        
        # Read back and verify
        for i, expected in enumerate(pattern):
            actual = self.jtag.read_mem(0x20000000 + i*4)
            if actual != expected:
                return f"FAIL: Memory error at {hex(0x20000000 + i*4)}"
        
        return "PASS: Memory controller OK"
    
    def test_thermal_behavior(self):
        """Run stress test and monitor temperature"""
        # Run CPU-intensive workload
        self.jtag.load_stress_test()
        self.jtag.run()
        
        max_temp = 0
        for _ in range(60):  # 60 second test
            time.sleep(1)
            temp = self.jtag.read_temperature()
            max_temp = max(max_temp, temp)
            if temp > 95:  # Thermal limit
                self.jtag.halt()
                return f"FAIL: Thermal limit exceeded ({temp}°C)"
        
        self.jtag.halt()
        return f"PASS: Max temp {max_temp}°C"
    
    def test_clock_speeds(self):
        """Verify processor can run at rated speeds"""
        rated_mhz = self.jtag.get_rated_clock()
        
        for freq in [rated_mhz, rated_mhz * 0.9, rated_mhz * 0.8]:
            self.jtag.set_clock(freq)
            if not self.verify_clock_stability(freq):
                return f"FAIL: Unstable at {freq} MHz"
        
        return f"PASS: Stable up to {rated_mhz} MHz"
    
    def calculate_grade(self):
        """Calculate overall grade based on test results"""
        failures = sum(1 for r in self.results.values() 
                      if "FAIL" in str(r))
        
        if failures == 0:
            return "A"
        elif failures <= 1:
            return "B"
        elif failures <= 2:
            return "C"
        else:
            return "F"
```

### 4.2 RAM Testing

#### 4.2.1 Test Equipment

- Memory test platform (DDR test board)
- SPD reader
- Power analyzer
- Temperature monitor

#### 4.2.2 Test Sequence

```python
class RAMModuleTest:
    """Test sequence for recycled RAM modules"""
    
    def __init__(self, test_platform):
        self.platform = test_platform
        self.results = {}
    
    def run_all_tests(self):
        """Execute complete RAM test suite"""
        tests = [
            self.test_spd_read,
            self.test_basic_rw,
            self.test_pattern_walking_ones,
            self.test_pattern_walking_zeros,
            self.test_address_lines,
            self.test_refresh,
            self.test_speed_grade,
        ]
        
        for test in tests:
            try:
                result = test()
                self.results[test.__name__] = result
            except Exception as e:
                self.results[test.__name__] = f"FAIL: {e}"
        
        return self.calculate_grade()
    
    def test_spd_read(self):
        """Read SPD EEPROM data"""
        spd = self.platform.read_spd()
        if spd is None:
            return "FAIL: SPD not readable"
        
        self.spd_data = spd
        return f"PASS: {spd['type']} {spd['size_mb']}MB {spd['speed_mhz']}MHz"
    
    def test_basic_rw(self):
        """Basic read/write test"""
        size_mb = self.spd_data['size_mb']
        test_size = min(64, size_mb)  # Test first 64MB
        
        # Write incremental pattern
        for addr in range(0, test_size * 1024 * 1024, 4):
            self.platform.write(addr, addr & 0xFFFFFFFF)
        
        # Verify
        errors = 0
        for addr in range(0, test_size * 1024 * 1024, 4):
            expected = addr & 0xFFFFFFFF
            actual = self.platform.read(addr)
            if actual != expected:
                errors += 1
                if errors > 10:  # Stop early if many errors
                    break
        
        if errors > 0:
            return f"FAIL: {errors} errors in basic R/W"
        return "PASS: Basic R/W OK"
    
    def test_pattern_walking_ones(self):
        """Walking ones test for data lines"""
        base_addr = 0x1000
        
        for bit in range(32):
            pattern = 1 << bit
            self.platform.write(base_addr, pattern)
            actual = self.platform.read(base_addr)
            if actual != pattern:
                return f"FAIL: Bit {bit} stuck"
        
        return "PASS: Walking ones OK"
    
    def test_pattern_walking_zeros(self):
        """Walking zeros test for data lines"""
        base_addr = 0x1000
        
        for bit in range(32):
            pattern = 0xFFFFFFFF ^ (1 << bit)
            self.platform.write(base_addr, pattern)
            actual = self.platform.read(base_addr)
            if actual != pattern:
                return f"FAIL: Bit {bit} stuck"
        
        return "PASS: Walking zeros OK"
    
    def test_address_lines(self):
        """Test address line integrity"""
        # Write unique pattern to power-of-2 addresses
        for bit in range(24):  # Up to 16MB
            addr = 1 << bit
            self.platform.write(addr, addr)
        
        # Verify no address aliasing
        for bit in range(24):
            addr = 1 << bit
            actual = self.platform.read(addr)
            if actual != addr:
                return f"FAIL: Address line A{bit} faulty"
        
        return "PASS: Address lines OK"
    
    def test_refresh(self):
        """Test data retention (refresh functionality)"""
        # Write pattern
        test_addr = 0x100000
        pattern = 0xA5A5A5A5
        self.platform.write(test_addr, pattern)
        
        # Wait 100ms (many refresh cycles)
        time.sleep(0.1)
        
        # Verify retention
        actual = self.platform.read(test_addr)
        if actual != pattern:
            return "FAIL: Refresh failure suspected"
        
        return "PASS: Refresh OK"
    
    def test_speed_grade(self):
        """Determine actual achievable speed"""
        rated_speed = self.spd_data['speed_mhz']
        
        achieved_speed = rated_speed
        for speed in [rated_speed, rated_speed - 100, rated_speed - 200]:
            self.platform.set_clock(speed)
            if self.quick_memtest():
                achieved_speed = speed
                break
            achieved_speed = speed - 100
        
        percentage = (achieved_speed / rated_speed) * 100
        
        if percentage >= 100:
            return f"PASS: Full speed ({rated_speed} MHz)"
        elif percentage >= 90:
            return f"PASS: {percentage:.0f}% speed ({achieved_speed} MHz)"
        else:
            return f"FAIL: Only {percentage:.0f}% speed ({achieved_speed} MHz)"
    
    def calculate_grade(self):
        """Calculate overall grade"""
        failures = sum(1 for r in self.results.values() 
                      if "FAIL" in str(r))
        
        if failures == 0:
            return "A"
        elif failures == 1:
            return "B"
        elif failures <= 2:
            return "C"
        else:
            return "F"
```

### 4.3 Extended Stress Testing

For Grade A certification, additional stress testing is required:

```python
def extended_stress_test(component, hours=72):
    """
    Extended burn-in test for production qualification
    
    Parameters:
    - component: Component under test
    - hours: Test duration (default 72h)
    
    Returns:
    - TestResult with detailed metrics
    """
    result = TestResult()
    start_time = time.time()
    
    while (time.time() - start_time) < hours * 3600:
        # Cycle through stress patterns
        stress_patterns = [
            all_zeros,
            all_ones,
            checkerboard,
            inverse_checkerboard,
            random_pattern,
        ]
        
        for pattern in stress_patterns:
            # Apply pattern
            component.apply_stress(pattern)
            
            # Monitor for errors
            errors = component.check_errors()
            result.log_errors(errors)
            
            # Check thermal limits
            temp = component.read_temperature()
            result.log_temperature(temp)
            
            if temp > component.thermal_limit:
                result.fail("Thermal limit exceeded")
                return result
            
            if errors > component.error_threshold:
                result.fail("Error threshold exceeded")
                return result
        
        # Log progress
        elapsed = (time.time() - start_time) / 3600
        result.log_progress(elapsed, hours)
    
    result.pass_test()
    return result
```

## 5. Inventory Management

### 5.1 Database Schema

```sql
CREATE TABLE components (
    id UUID PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    part_number VARCHAR(100),
    manufacturer VARCHAR(100),
    source_device VARCHAR(200),
    source_channel VARCHAR(50),
    acquisition_date DATE,
    test_date DATE,
    grade CHAR(1),
    status VARCHAR(20),
    location VARCHAR(50),
    notes TEXT
);

CREATE TABLE test_results (
    id UUID PRIMARY KEY,
    component_id UUID REFERENCES components(id),
    test_type VARCHAR(50),
    test_date TIMESTAMP,
    result VARCHAR(20),
    details JSONB,
    tester_id VARCHAR(50)
);

CREATE TABLE usage_history (
    id UUID PRIMARY KEY,
    component_id UUID REFERENCES components(id),
    used_in_system VARCHAR(100),
    install_date DATE,
    remove_date DATE,
    reason VARCHAR(200)
);
```

### 5.2 Barcode/QR Labeling

Each tested component receives a unique label:

```
┌─────────────────────────────────┐
│  PENTARY COMPONENT              │
│  ┌─────────┐                    │
│  │ QR CODE │  ARM-A53-00001-A   │
│  │         │  Grade: A          │
│  │         │  Date: 2026-01-08  │
│  └─────────┘                    │
└─────────────────────────────────┘
```

## 6. Quality Control

### 6.1 Acceptance Criteria

**Incoming Inspection:**
- Visual inspection for damage (100%)
- SPD/ID verification (100%)
- Quick functional test (100%)

**Detailed Testing:**
- Full test suite (100% of items passing quick test)
- Stress testing (10% sample, or 100% for Grade A)

**Outgoing Quality:**
- Final visual inspection
- Verification of labeling
- Packaging check

### 6.2 Failure Analysis

When components fail testing:

1. **Document failure mode**
   - Which test failed
   - Error details
   - Environmental conditions

2. **Root cause analysis** (sampling)
   - Physical inspection
   - Cross-reference with source
   - Identify patterns

3. **Feedback loop**
   - Adjust sourcing criteria
   - Update test procedures
   - Improve extraction methods

## 7. Environmental and Safety

### 7.1 Hazardous Materials

| Material | Location | Handling |
|----------|----------|----------|
| Lead (Pb) | Older solder | Ventilation, gloves |
| Brominated compounds | PCB flame retardants | Avoid heating >300°C |
| Lithium | Batteries | Discharge, isolate |
| Beryllium | Some heatsinks | Avoid grinding |

### 7.2 Safety Equipment

Required PPE:
- Safety glasses
- ESD wrist strap
- Heat-resistant gloves (for desoldering)
- Fume extractor

### 7.3 Waste Handling

Components that fail testing:
1. Segregate by material type
2. Document quantities
3. Send to certified recycler
4. Maintain chain of custody records

## 8. Cost Analysis

### 8.1 Per-Component Economics

**ARM Processor (Cortex-A53):**
| Item | Cost |
|------|------|
| Acquisition (in phone) | $5-15 |
| Extraction labor | $3-5 |
| Testing | $2-3 |
| Total (Grade A) | $10-23 |
| New equivalent | $15-40 |
| **Savings** | **30-50%** |

**DDR4 RAM (8GB):**
| Item | Cost |
|------|------|
| Acquisition | $3-8 |
| Testing | $1-2 |
| Total (Grade A) | $4-10 |
| New equivalent | $20-35 |
| **Savings** | **70-80%** |

### 8.2 Volume Discounts

| Monthly Volume | Discount |
|----------------|----------|
| < 100 units | Base price |
| 100-500 units | 10% |
| 500-1000 units | 20% |
| > 1000 units | 30% |

## 9. Scaling Roadmap

### Phase 1: Pilot (Months 1-6)
- Manual extraction and testing
- 100 components/month capacity
- Refine procedures

### Phase 2: Semi-Automated (Months 7-12)
- Automated testing fixtures
- 500 components/month capacity
- Establish supply partnerships

### Phase 3: Scaled Operation (Year 2+)
- Automated extraction stations
- 2000+ components/month
- Multiple test lines
- Regional collection centers

## Appendix: Supplier Agreement Template

```
RECYCLED COMPONENT SUPPLY AGREEMENT

Between: [Pentary Computing, Inc.] ("Buyer")
And: [Supplier Name] ("Supplier")

1. COMPONENT SPECIFICATIONS
   Supplier agrees to provide:
   - Component type: [e.g., ARM Cortex-A53 processors]
   - Minimum quantity: [X] units per month
   - Source devices: [Smartphones, tablets, etc.]
   
2. QUALITY REQUIREMENTS
   - Minimum [90]% functional yield after Buyer testing
   - Components must not show physical damage
   - Original part numbers must be legible
   
3. PRICING
   - Per unit price: $[X.XX]
   - Volume discounts per schedule attached
   
4. DELIVERY
   - Monthly delivery schedule
   - Packaging requirements: Anti-static, labeled
   
5. RETURNS
   - Non-functional units may be returned
   - Credit issued within 30 days
   
6. ENVIRONMENTAL COMPLIANCE
   - Supplier certifies compliance with applicable
     e-waste regulations
   - Documentation provided upon request

Signed: _______________ Date: ___________
```
