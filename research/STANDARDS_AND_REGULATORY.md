# Pentary Computing: Standards and Regulatory Considerations

This document outlines standards compliance, regulatory requirements, and certification considerations for Pentary computing products.

---

## Executive Summary

Pentary computing products must comply with multiple standards and regulations:

| Category | Key Standards | Priority | Timeline |
|----------|---------------|----------|----------|
| **Safety** | UL, CE, FCC | Critical | Before sale |
| **EMC** | FCC Part 15, EN 55032 | Critical | Before sale |
| **Environmental** | RoHS, REACH | Critical | Before production |
| **Quality** | ISO 9001 | High | Before volume |
| **AI/ML Specific** | EU AI Act, NIST AI RMF | Medium | Emerging |
| **Export Control** | EAR, ITAR | High | Before export |

---

## 1. Electronic Safety Standards

### 1.1 North America

#### UL Standards
| Standard | Scope | Requirement |
|----------|-------|-------------|
| **UL 62368-1** | IT equipment safety | Mandatory for US/Canada |
| **UL 94** | Flammability | Component requirement |
| **UL 60950** | Legacy IT (obsolete) | Reference only |

#### FCC Certification
| Class | Application | Requirement |
|-------|-------------|-------------|
| **Class A** | Commercial/industrial | Verification |
| **Class B** | Residential | Certification |

**Pentary Products:**
- Pentary Monolith (server): Class A
- Pentary Deck (consumer): Class B
- Pentary Reflex (industrial): Class A

### 1.2 European Union

#### CE Marking
Required directives:
| Directive | Scope | Standard |
|-----------|-------|----------|
| **LVD** 2014/35/EU | Safety | EN 62368-1 |
| **EMC** 2014/30/EU | Electromagnetic | EN 55032/35 |
| **RoHS** 2011/65/EU | Hazardous substances | EN 50581 |
| **WEEE** 2012/19/EU | Waste disposal | N/A |

#### Technical File Requirements
- Risk assessment
- Design documentation
- Test reports
- Declaration of Conformity

### 1.3 International

| Region | Mark | Standard |
|--------|------|----------|
| China | CCC | GB 4943 |
| Japan | PSE | J62368-1 |
| Korea | KC | K62368-1 |
| Australia | RCM | AS/NZS 62368.1 |

---

## 2. Electromagnetic Compatibility (EMC)

### 2.1 Emission Standards

| Region | Standard | Limits |
|--------|----------|--------|
| US | FCC Part 15 | Class A/B |
| EU | EN 55032 | Class A/B |
| International | CISPR 32 | Class A/B |

**Pentary EMC Considerations:**
- High-speed digital signals (multi-GHz)
- Memristor switching noise
- Power supply harmonics
- Crossbar crosstalk

### 2.2 Immunity Standards

| Standard | Test | Level |
|----------|------|-------|
| EN 61000-4-2 | ESD | ±4kV contact, ±8kV air |
| EN 61000-4-3 | Radiated immunity | 3-10 V/m |
| EN 61000-4-4 | EFT | 2 kV |
| EN 61000-4-5 | Surge | 1-2 kV |
| EN 61000-4-6 | Conducted immunity | 3-10 V |

### 2.3 EMC Design Guidelines

**Pentary-Specific EMC Considerations:**

```
EMC Design Checklist:

□ Multi-layer PCB (minimum 6 layers)
  ├── Layer 1: Signal
  ├── Layer 2: Ground plane (unbroken)
  ├── Layer 3: Signal
  ├── Layer 4: Power plane
  ├── Layer 5: Signal
  └── Layer 6: Ground plane

□ Clock distribution
  ├── Spread-spectrum clocking for EMI reduction
  ├── Matched trace lengths
  └── Differential signaling where possible

□ Power supply filtering
  ├── Input filter for conducted emissions
  ├── Decoupling at each power domain
  └── Ferrite beads for high-frequency noise

□ Memristor array shielding
  ├── Local ground plane under array
  ├── Guard traces for sensitive signals
  └── Crosstalk minimization
```

---

## 3. Environmental Regulations

### 3.1 RoHS Compliance

**Restricted Substances:**
| Substance | Limit |
|-----------|-------|
| Lead (Pb) | 0.1% |
| Mercury (Hg) | 0.1% |
| Cadmium (Cd) | 0.01% |
| Hexavalent chromium | 0.1% |
| PBB | 0.1% |
| PBDE | 0.1% |
| DEHP, BBP, DBP, DIBP | 0.1% each |

**Pentary RoHS Considerations:**
- Memristor materials (verify Pb-free)
- Solder paste selection
- Component qualification

### 3.2 REACH Compliance

**Requirements:**
- SVHC (Substances of Very High Concern) reporting
- Supply chain documentation
- Annual updates

### 3.3 Conflict Minerals

**Dodd-Frank Act requirements:**
- 3TG reporting (Tin, Tantalum, Tungsten, Gold)
- Supply chain due diligence
- Annual disclosure

### 3.4 WEEE Compliance

**Requirements:**
- Product registration
- Collection schemes
- Recycling targets
- Producer responsibility

**Design for Recyclability:**
- Material identification marking
- Easy disassembly
- Minimize material types

---

## 4. Quality Management Standards

### 4.1 ISO 9001:2015

**Requirements for Pentary:**
| Clause | Requirement | Implementation |
|--------|-------------|----------------|
| 4 | Context | Define scope for AI hardware |
| 5 | Leadership | Quality policy |
| 6 | Planning | Risk-based thinking |
| 7 | Support | Resources, competence |
| 8 | Operation | Design control, production |
| 9 | Performance | Monitoring, measurement |
| 10 | Improvement | Continuous improvement |

### 4.2 Automotive (if applicable)

**IATF 16949:**
- Required for automotive customers
- Adds automotive-specific requirements
- PPAP documentation

**AEC-Q100:**
- Automotive IC qualification
- Stress test requirements
- Reliability targets

### 4.3 Aerospace/Defense (if applicable)

**AS9100:**
- Aerospace quality management
- Configuration management
- Risk management

---

## 5. AI-Specific Regulations

### 5.1 EU AI Act

**Risk Categories:**
| Risk Level | Examples | Requirements |
|------------|----------|--------------|
| Unacceptable | Social scoring | Prohibited |
| High | Critical infrastructure, hiring | Conformity assessment |
| Limited | Chatbots, emotion recognition | Transparency |
| Minimal | Spam filters, games | None |

**Pentary Product Classification:**
- Pentary Monolith (inference): Likely "Limited" or "Minimal"
- If used in high-risk applications: "High" requirements apply

**High-Risk Requirements (if applicable):**
1. Risk management system
2. Data governance
3. Technical documentation
4. Record-keeping
5. Transparency
6. Human oversight
7. Accuracy, robustness, cybersecurity

### 5.2 NIST AI Risk Management Framework

**Core Functions:**
| Function | Subfunctions |
|----------|--------------|
| **GOVERN** | Policies, accountability |
| **MAP** | Context, risk framing |
| **MEASURE** | Assessment, metrics |
| **MANAGE** | Prioritize, respond |

**Voluntary but increasingly referenced.**

### 5.3 Algorithmic Accountability

**Considerations:**
- Bias detection in quantization
- Explainability requirements
- Audit trails
- Impact assessments

---

## 6. Export Control Regulations

### 6.1 US Export Administration Regulations (EAR)

**ECCN Classification:**
| Category | Description | Pentary Relevance |
|----------|-------------|-------------------|
| 3A001 | Electronics | IC classification |
| 3A991 | Electronics (EAR99) | Less controlled |
| 4A003 | Computers | Performance-based |
| 4D001 | Software | Development tools |
| 5A002 | Security | Encryption |

**Performance Thresholds:**
- Composite Theoretical Performance (CTP)
- Adjusted Peak Performance (APP)
- Memory bandwidth limits

**Pentary Classification (Preliminary):**
- Likely EAR99 or 4A994 (if no encryption)
- Higher control if used in specific applications

### 6.2 Entity List Considerations

**Restrictions on sales to:**
- Listed entities in China, Russia, etc.
- End-use restrictions
- Military applications

### 6.3 ITAR (If Defense-Related)

**Requirements:**
- Registration with DDTC
- Export licenses
- TAA for technical data
- ITAR-compliant supply chain

---

## 7. Industry-Specific Standards

### 7.1 Data Center

| Standard | Scope | Requirement |
|----------|-------|-------------|
| **ASHRAE A3/A4** | Environmental | Operating conditions |
| **OCP** | Open compute | Design specs |
| **DMTF Redfish** | Management | BMC interface |

### 7.2 Edge/IoT

| Standard | Scope | Requirement |
|----------|-------|-------------|
| **IP67/IP68** | Ingress protection | Environmental |
| **MIL-STD-810** | Ruggedness | Military/industrial |
| **IEC 61131** | Industrial control | PLC integration |

### 7.3 Automotive

| Standard | Scope | Requirement |
|----------|-------|-------------|
| **ISO 26262** | Functional safety | ASIL levels |
| **ASPICE** | Process maturity | Development quality |
| **ISO 21434** | Cybersecurity | Automotive security |

### 7.4 Medical (If Applicable)

| Standard | Scope | Requirement |
|----------|-------|-------------|
| **IEC 62304** | Software lifecycle | Medical device software |
| **IEC 60601** | Safety | Medical electrical |
| **FDA 21 CFR Part 11** | Electronic records | US compliance |

---

## 8. Cybersecurity Standards

### 8.1 Hardware Security

| Standard | Scope |
|----------|-------|
| **FIPS 140-3** | Cryptographic modules |
| **Common Criteria** | IT security evaluation |
| **TCG (TPM)** | Trusted platform |

### 8.2 AI Security

**Considerations:**
- Model extraction attacks
- Adversarial inputs
- Data poisoning
- Side-channel attacks

**Pentary Security Features:**
- Weight obfuscation
- Integrity verification
- Secure boot
- Hardware root of trust

---

## 9. Benchmarking Standards

### 9.1 MLPerf

**Benchmarks:**
| Benchmark | Description | Metric |
|-----------|-------------|--------|
| MLPerf Training | Training performance | Time to accuracy |
| MLPerf Inference | Inference performance | Throughput, latency |
| MLPerf Mobile | Mobile inference | Throughput/watt |
| MLPerf Tiny | Edge inference | Energy/inference |

**Pentary Participation Strategy:**
1. Target MLPerf Inference first
2. Submit to Edge category
3. Publish power efficiency metrics

### 9.2 Other Benchmarks

| Benchmark | Focus |
|-----------|-------|
| **SPEC ML** | Workload-based |
| **AI Benchmark** | Mobile devices |
| **MLMark** | Embedded ML |

---

## 10. Certification Process

### 10.1 Timeline Planning

```
Certification Timeline (12-18 months before launch):

Month -18: │ Begin standards review
           │ Engage certification lab
           │
Month -15: │ Pre-compliance testing
           │ Design modifications
           │
Month -12: │ Submit for EMC testing
           │ Submit for safety evaluation
           │
Month -9:  │ EMC testing complete
           │ Safety certification in progress
           │
Month -6:  │ Receive certifications
           │ Finalize technical file
           │
Month -3:  │ Final compliance verification
           │ Production certification
           │
Month 0:   │ Product launch
```

### 10.2 Certification Costs

| Certification | Est. Cost | Timeline |
|---------------|-----------|----------|
| FCC Class B | $15K-$25K | 4-6 weeks |
| CE Mark (Full) | $30K-$50K | 8-12 weeks |
| UL Listing | $20K-$40K | 8-16 weeks |
| China CCC | $15K-$30K | 12-16 weeks |
| Total First Product | **$100K-$200K** | 6-12 months |

### 10.3 Certification Bodies

| Region | Bodies |
|--------|--------|
| US | UL, Intertek, TÜV |
| EU | TÜV, BSI, SGS |
| Asia | SGS, Intertek, local bodies |

---

## 11. Documentation Requirements

### 11.1 Technical Documentation

**Required Documents:**
1. System design description
2. Schematic diagrams
3. PCB layout files
4. Bill of materials
5. Test reports
6. Risk assessment
7. User manual

### 11.2 Declaration of Conformity

**EU DoC Requirements:**
- Manufacturer identification
- Product identification
- Directives referenced
- Standards applied
- Authorized representative
- Date and signature

### 11.3 Record Retention

| Document | Retention Period |
|----------|------------------|
| Design records | 10 years |
| Test reports | 10 years |
| DoC | 10 years after last unit |
| Quality records | 7 years |

---

## 12. Compliance Roadmap

### 12.1 Phase 1: Development Samples

**Focus:** Pre-compliance testing
- EMC scan testing
- Safety review
- Power integrity
- Thermal analysis

### 12.2 Phase 2: Engineering Prototypes

**Focus:** Formal compliance testing
- EMC certification
- Safety evaluation
- Environmental testing

### 12.3 Phase 3: Production

**Focus:** Volume certification
- Production line approval
- Quality system audit
- Ongoing compliance

### 12.4 Compliance Checklist

```
Pre-Launch Compliance Checklist:

Safety:
□ UL 62368-1 certification
□ CE marking (LVD)
□ Other regional certifications

EMC:
□ FCC certification
□ CE marking (EMC)
□ Other regional certifications

Environmental:
□ RoHS compliance declaration
□ REACH SVHC assessment
□ Conflict minerals report
□ WEEE registration

Quality:
□ ISO 9001 (if required)
□ Production quality plan
□ Supply chain qualification

Export:
□ ECCN classification
□ License determination
□ Compliance program

AI-Specific:
□ EU AI Act assessment
□ Risk classification
□ Documentation requirements
```

---

## 13. Recommendations

### 13.1 Immediate Actions

1. **Engage compliance consultant** early in design
2. **Design for compliance** from the start
3. **Budget appropriately** ($100K-$200K for first product)
4. **Select certification lab** partner

### 13.2 Design Guidelines

1. **EMC:** Plan PCB stackup for EMI compliance
2. **Safety:** Include required isolation and protection
3. **Thermal:** Design for operating temperature range
4. **Environmental:** Select RoHS-compliant components

### 13.3 Ongoing Compliance

1. **Monitor regulatory changes** (especially EU AI Act)
2. **Update certifications** for design changes
3. **Maintain technical files** current
4. **Train team** on compliance requirements

---

**Document Version**: 1.0  
**Created**: January 2026  
**Status**: Regulatory framework defined  
**Next Update**: After detailed product specification
