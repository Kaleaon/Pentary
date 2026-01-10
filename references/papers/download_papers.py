#!/usr/bin/env python3
"""
Automated Paper Download Script for Pentary Research
Downloads papers from arXiv and other sources
"""

import os
import json
import subprocess
from datetime import datetime

# Create directories
os.makedirs("neuromorphic", exist_ok=True)
os.makedirs("infinite_context", exist_ok=True)

def download_arxiv_paper(arxiv_id, category, filename=None):
    """Download paper from arXiv"""
    if filename is None:
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
    
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = os.path.join(category, filename)
    
    try:
        subprocess.run(["wget", "-q", "-O", output_path, url], check=True)
        print(f"✓ Downloaded: {filename}")
        return True
    except:
        print(f"✗ Failed: {filename}")
        return False

# Extended paper collection with download information
papers_to_download = [
    # Neuromorphic Computing - Core Papers
    {"arxiv_id": "2410.23639", "category": "neuromorphic", "title": "Brain_Computer_Interfaces"},
    {"arxiv_id": "2408.03884", "category": "neuromorphic", "title": "Quantum_Neuromorphic_Computing"},
    {"arxiv_id": "2507.18139", "category": "neuromorphic", "title": "Embodied_Intelligence"},
    {"arxiv_id": "2410.09218", "category": "neuromorphic", "title": "Continual_Learning_Neuromorphic"},
    {"arxiv_id": "2507.17886", "category": "neuromorphic", "title": "Time_Space_Energy_Scaling"},
    {"arxiv_id": "2510.06721", "category": "neuromorphic", "title": "Neuromorphic_Overview"},
    {"arxiv_id": "2507.10722", "category": "neuromorphic", "title": "Bridging_Brains_Machines"},
    
    # Spiking Neural Networks
    {"arxiv_id": "2510.27379", "category": "neuromorphic", "title": "SNN_Future_Brain_Inspired"},
    {"arxiv_id": "2402.01782", "category": "neuromorphic", "title": "Benchmarking_SNN"},
    {"arxiv_id": "2510.20997", "category": "neuromorphic", "title": "SNN_Binary_Classification"},
    {"arxiv_id": "2410.02249", "category": "neuromorphic", "title": "SNN_Event_Stream"},
    {"arxiv_id": "2510.14235", "category": "neuromorphic", "title": "SNN_Architecture_Search"},
    {"arxiv_id": "2505.13622", "category": "neuromorphic", "title": "SNN_Random_Architecture"},
    {"arxiv_id": "2509.21920", "category": "neuromorphic", "title": "SNN_Universal_Computation"},
    
    # Neuromorphic Hardware
    {"arxiv_id": "2408.14680", "category": "neuromorphic", "title": "Memristor_Neural_Networks"},
    {"arxiv_id": "2503.13386", "category": "neuromorphic", "title": "Microfluidic_Memristors"},
    {"arxiv_id": "2407.13410", "category": "neuromorphic", "title": "Neuromorphic_Circuit_Simulation"},
    {"arxiv_id": "2504.18694", "category": "neuromorphic", "title": "Quantum_Memristor"},
    {"arxiv_id": "2409.10887", "category": "neuromorphic", "title": "Contrastive_Learning_Memristor"},
    {"arxiv_id": "2512.07602", "category": "neuromorphic", "title": "Algorithm_Hardware_Codesign"},
    {"arxiv_id": "2511.03747", "category": "neuromorphic", "title": "OpenMENA_Memristor"},
    {"arxiv_id": "2509.04506", "category": "neuromorphic", "title": "Memristor_Space_Applications"},
    {"arxiv_id": "2505.12960", "category": "neuromorphic", "title": "Superlinear_Memristor"},
    {"arxiv_id": "2510.14120", "category": "neuromorphic", "title": "Laser_Fault_Injection"},
    
    # Brain-Inspired Learning
    {"arxiv_id": "2511.04455", "category": "neuromorphic", "title": "Brain_Inspired_Learning_Survey"},
    {"arxiv_id": "2308.07870", "category": "neuromorphic", "title": "Predictive_Coding"},
    {"arxiv_id": "2502.20411", "category": "neuromorphic", "title": "Backprop_Free_SNN"},
    {"arxiv_id": "2501.09238", "category": "neuromorphic", "title": "Mono_Forward"},
    {"arxiv_id": "2304.11042", "category": "neuromorphic", "title": "Backprop_Free_Training"},
    
    # Photonic Neuromorphic
    {"arxiv_id": "2408.02685", "category": "neuromorphic", "title": "Photonic_Neural_Networks"},
    {"arxiv_id": "2507.20866", "category": "neuromorphic", "title": "Photonic_Spiking_Memory"},
    {"arxiv_id": "2506.14575", "category": "neuromorphic", "title": "Integrated_Photonic_DNN"},
    {"arxiv_id": "2509.01262", "category": "neuromorphic", "title": "Integrated_Photonic_Neuromorphic"},
    {"arxiv_id": "2311.09767", "category": "neuromorphic", "title": "Photonics_Neuromorphic_Fundamentals"},
    {"arxiv_id": "2401.16515", "category": "neuromorphic", "title": "Electro_Optic_Analog"},
    
    # Energy Efficient Computing
    {"arxiv_id": "2508.09163", "category": "neuromorphic", "title": "Stochastic_Computing"},
    {"arxiv_id": "2509.15097", "category": "neuromorphic", "title": "Energy_Efficient_Hierarchical"},
    {"arxiv_id": "2509.00764", "category": "neuromorphic", "title": "Low_Power_Multiplier"},
    {"arxiv_id": "2403.08151", "category": "neuromorphic", "title": "Energy_Consumption_Measurement"},
    {"arxiv_id": "2508.20720", "category": "neuromorphic", "title": "Energy_Efficient_Generative"},
    
    # Temporal Coding
    {"arxiv_id": "2501.14484", "category": "neuromorphic", "title": "Temporal_Coding"},
    {"arxiv_id": "2407.08744", "category": "neuromorphic", "title": "Temporal_Single_Spike"},
    {"arxiv_id": "2510.25993", "category": "neuromorphic", "title": "Predictive_Coding_Online"},
    {"arxiv_id": "2507.16043", "category": "neuromorphic", "title": "Spike_Timing_Gradients"},
    {"arxiv_id": "2508.20392", "category": "neuromorphic", "title": "Ultra_Low_Latency_SNN"},
    {"arxiv_id": "2502.09449", "category": "neuromorphic", "title": "SNN_Temporal_Processing"},
    
    # Plasticity and Learning Rules
    {"arxiv_id": "2506.19377", "category": "neuromorphic", "title": "STDP_Unified_Platform"},
    {"arxiv_id": "2507.21474", "category": "neuromorphic", "title": "Hebbian_Memory_Augmented"},
    {"arxiv_id": "2407.17305", "category": "neuromorphic", "title": "Continual_Hebbian"},
    {"arxiv_id": "2512.09366", "category": "neuromorphic", "title": "Meta_Learning_Plasticity"},
    {"arxiv_id": "2504.05341", "category": "neuromorphic", "title": "Three_Factor_Learning"},
    {"arxiv_id": "2505.18069", "category": "neuromorphic", "title": "Hebbian_Dynamics"},
    {"arxiv_id": "2511.14691", "category": "neuromorphic", "title": "Spiking_Transformer"},
    {"arxiv_id": "2506.14984", "category": "neuromorphic", "title": "STDP_Extension"},
    {"arxiv_id": "2505.10272", "category": "neuromorphic", "title": "STDP_Gradient_Descent"},
    
    # Event-Based Vision
    {"arxiv_id": "2405.18880", "category": "neuromorphic", "title": "Event_Based_Augmentation"},
    {"arxiv_id": "2412.15021", "category": "neuromorphic", "title": "Event_Based_Backprop"},
    {"arxiv_id": "2504.08588", "category": "neuromorphic", "title": "Neuromorphic_Vision_Hardware"},
    {"arxiv_id": "2508.19806", "category": "neuromorphic", "title": "Sparse_Spatiotemporal"},
    {"arxiv_id": "2511.20175", "category": "neuromorphic", "title": "Event_Based_Pupil_Tracking"},
    {"arxiv_id": "2411.13108", "category": "neuromorphic", "title": "Event_Based_Suitability"},
    {"arxiv_id": "2504.15371", "category": "neuromorphic", "title": "Event2Vec"},
    {"arxiv_id": "2407.00931", "category": "neuromorphic", "title": "Event_Based_Physics"},
    
    # Biological Neural Networks
    {"arxiv_id": "2501.18018", "category": "neuromorphic", "title": "Neuroscience_Inspired_ANN"},
    {"arxiv_id": "2402.12796", "category": "neuromorphic", "title": "Neural_Codes_Dynamics"},
    {"arxiv_id": "2509.23896", "category": "neuromorphic", "title": "NeuroAI_Synthetic_Biological"},
    {"arxiv_id": "2409.09125", "category": "neuromorphic", "title": "Quantum_Biological_Neurons"},
    
    # Analog Computing
    {"arxiv_id": "2409.14918", "category": "neuromorphic", "title": "Analog_Digital_Simulation"},
    {"arxiv_id": "2502.20381", "category": "neuromorphic", "title": "Analog_Signal_Processing"},
    {"arxiv_id": "2401.16840", "category": "neuromorphic", "title": "Scalable_Network_Emulation"},
    {"arxiv_id": "2412.14029", "category": "neuromorphic", "title": "Temperature_Resilient_Analog"},
    {"arxiv_id": "2412.09010", "category": "neuromorphic", "title": "Harnessing_Nonidealities"},
    
    # Processing-in-Memory
    {"arxiv_id": "2512.00096", "category": "neuromorphic", "title": "PIM_Modeling_Simulation"},
    {"arxiv_id": "2502.21259", "category": "neuromorphic", "title": "PIM_System_Performance"},
    {"arxiv_id": "2412.19275", "category": "neuromorphic", "title": "Memory_Centric_Computing"},
    {"arxiv_id": "2510.07719", "category": "neuromorphic", "title": "DL_PIM"},
    {"arxiv_id": "2509.12993", "category": "neuromorphic", "title": "HPIM_Heterogeneous"},
    {"arxiv_id": "2502.07578", "category": "neuromorphic", "title": "PIM_GPU_Free"},
    
    # Edge AI and TinyML
    {"arxiv_id": "2409.07114", "category": "neuromorphic", "title": "Continual_TinyML"},
    {"arxiv_id": "2510.01439", "category": "neuromorphic", "title": "Edge_AI_Evolution"},
    {"arxiv_id": "2506.18927", "category": "neuromorphic", "title": "Tiny_Deep_Learning"},
    {"arxiv_id": "2503.06027", "category": "neuromorphic", "title": "On_Device_AI_Survey"},
    {"arxiv_id": "2403.19076", "category": "neuromorphic", "title": "Tiny_ML_Progress"},
    {"arxiv_id": "2405.07601", "category": "neuromorphic", "title": "On_Device_Online_Learning"},
    {"arxiv_id": "2502.17788", "category": "neuromorphic", "title": "On_Device_Edge_Learning"},
    
    # Neural Architecture Search
    {"arxiv_id": "2502.20422", "category": "neuromorphic", "title": "Self_Evolution_NAS"},
    {"arxiv_id": "2302.04406", "category": "neuromorphic", "title": "NAS_Shared_Weights"},
    {"arxiv_id": "2403.07591", "category": "neuromorphic", "title": "Training_Free_NAS"},
    {"arxiv_id": "2502.03553", "category": "neuromorphic", "title": "Efficient_Global_NAS"},
    {"arxiv_id": "2507.13485", "category": "neuromorphic", "title": "NAS_Bio_Inspired"},
    {"arxiv_id": "2510.01472", "category": "neuromorphic", "title": "PEL_NAS"},
    {"arxiv_id": "2305.05351", "category": "neuromorphic", "title": "GPT_NAS"},
    
    # Continual Learning
    {"arxiv_id": "2404.14829", "category": "neuromorphic", "title": "Revisiting_Continual_Learning"},
    {"arxiv_id": "1802.07569", "category": "neuromorphic", "title": "Continual_Lifelong_Review"},
    {"arxiv_id": "1910.02718", "category": "neuromorphic", "title": "Continual_Learning_NN"},
    {"arxiv_id": "2506.03320", "category": "neuromorphic", "title": "Future_Continual_Learning"},
    {"arxiv_id": "2202.10821", "category": "neuromorphic", "title": "Increasing_Depth_Lifelong"},
    {"arxiv_id": "2511.12793", "category": "neuromorphic", "title": "Neuro_Logic_Lifelong"},
    
    # Compression and Quantization
    {"arxiv_id": "2509.04244", "category": "neuromorphic", "title": "Pruning_Quantization_Integration"},
    {"arxiv_id": "2506.20084", "category": "neuromorphic", "title": "Joint_Quantization_Pruning"},
    {"arxiv_id": "2511.19495", "category": "neuromorphic", "title": "Compression_Ordering_LLM"},
    {"arxiv_id": "2510.22058", "category": "neuromorphic", "title": "Pruning_Quantization_GNN"},
    {"arxiv_id": "2412.18184", "category": "neuromorphic", "title": "Unified_Stochastic_Framework"},
    {"arxiv_id": "2307.02973", "category": "neuromorphic", "title": "Pruning_vs_Quantization"},
    
    # INFINITE CONTEXT PAPERS
    
    # Long Context Transformers
    {"arxiv_id": "2502.12962", "category": "infinite_context", "title": "Infinite_Retrieval"},
    {"arxiv_id": "2404.07143", "category": "infinite_context", "title": "Infini_Attention"},
    {"arxiv_id": "2503.22196", "category": "infinite_context", "title": "Memory_Efficient_Edge"},
    {"arxiv_id": "2407.15176", "category": "infinite_context", "title": "ReAttention"},
    
    # Memory-Augmented Networks
    {"arxiv_id": "2312.06141", "category": "infinite_context", "title": "Memory_Augmented_Survey"},
    {"arxiv_id": "2508.10824", "category": "infinite_context", "title": "Memory_Augmented_Transformers"},
    {"arxiv_id": "2507.21474", "category": "infinite_context", "title": "Hebbian_Memory_RNN"},
    {"arxiv_id": "2510.27246", "category": "infinite_context", "title": "Long_Term_Memory_LLM"},
    {"arxiv_id": "2310.10909", "category": "infinite_context", "title": "Heterogenous_Memory"},
    {"arxiv_id": "2410.03154", "category": "infinite_context", "title": "Learnability_Memory_RNN"},
    
    # State Space Models
    {"arxiv_id": "2411.11843", "category": "infinite_context", "title": "Bi_Mamba"},
    {"arxiv_id": "2501.16295", "category": "infinite_context", "title": "Mixture_of_Mamba"},
    {"arxiv_id": "2503.18970", "category": "infinite_context", "title": "SSM_Comprehensive_Survey"},
    {"arxiv_id": "2412.06148", "category": "infinite_context", "title": "Computational_Limits_Mamba"},
    {"arxiv_id": "2404.16112", "category": "infinite_context", "title": "Mamba_360_Survey"},
    {"arxiv_id": "2410.15091", "category": "infinite_context", "title": "Spatial_Mamba"},
    {"arxiv_id": "2407.13772", "category": "infinite_context", "title": "GroupMamba"},
    {"arxiv_id": "2312.00752", "category": "infinite_context", "title": "Mamba_Original"},
    
    # RAG Papers
    {"arxiv_id": "2501.00309", "category": "infinite_context", "title": "GraphRAG"},
    {"arxiv_id": "2410.05779", "category": "infinite_context", "title": "LightRAG"},
    {"arxiv_id": "2411.06037", "category": "infinite_context", "title": "Sufficient_Context_RAG"},
    {"arxiv_id": "2409.14924", "category": "infinite_context", "title": "RAG_and_Beyond"},
    {"arxiv_id": "2410.12837", "category": "infinite_context", "title": "RAG_Comprehensive_Survey"},
    {"arxiv_id": "2506.00054", "category": "infinite_context", "title": "RAG_Survey_2025"},
    {"arxiv_id": "2507.09477", "category": "infinite_context", "title": "Agentic_RAG"},
    {"arxiv_id": "2407.16833", "category": "infinite_context", "title": "RAG_vs_Long_Context"},
    
    # Context Compression
    {"arxiv_id": "2509.09199", "category": "infinite_context", "title": "CCF_Context_Compression"},
    {"arxiv_id": "2510.08907", "category": "infinite_context", "title": "Autoencoding_Free_Compression"},
    {"arxiv_id": "2511.18832", "category": "infinite_context", "title": "AMR_Conceptual_Entropy"},
    {"arxiv_id": "2505.23277", "category": "infinite_context", "title": "Sentinel_Attention_Probing"},
    {"arxiv_id": "2508.08514", "category": "infinite_context", "title": "DeCAL_Tokenwise"},
    {"arxiv_id": "2406.13618", "category": "infinite_context", "title": "Lightning_Fast_Compression"},
    
    # Recurrent Memory
    {"arxiv_id": "2505.07793", "category": "infinite_context", "title": "Overflow_Prevention_RNN"},
    {"arxiv_id": "2509.23040", "category": "infinite_context", "title": "Revisitable_Memory"},
    {"arxiv_id": "2407.04841", "category": "infinite_context", "title": "Associative_Recurrent_Memory"},
    {"arxiv_id": "2507.00453", "category": "infinite_context", "title": "Recurrent_Memory_Chunked"},
    {"arxiv_id": "2509.09505", "category": "infinite_context", "title": "Long_Context_Agentic"},
    
    # Attention Alternatives
    {"arxiv_id": "2412.00359", "category": "infinite_context", "title": "Attention_Alternatives"},
    {"arxiv_id": "2510.05364", "category": "infinite_context", "title": "End_of_Transformers"},
    {"arxiv_id": "2507.19595", "category": "infinite_context", "title": "Efficient_Attention_Survey"},
    {"arxiv_id": "2410.11842", "category": "infinite_context", "title": "MoH_Mixture_of_Head"},
    {"arxiv_id": "2311.10642", "category": "infinite_context", "title": "Shallow_Feed_Forward"},
    {"arxiv_id": "2509.20942", "category": "infinite_context", "title": "Attention_Degeneration"},
    {"arxiv_id": "2512.03377", "category": "infinite_context", "title": "Nexus_Higher_Order"},
    {"arxiv_id": "2410.04271", "category": "infinite_context", "title": "Subquadratic_Limitations"},
    
    # World Models
    {"arxiv_id": "2509.21797", "category": "infinite_context", "title": "Mixture_of_World_Models"},
    {"arxiv_id": "2503.18938", "category": "infinite_context", "title": "Adaptable_World_Models"},
    {"arxiv_id": "2511.11011", "category": "infinite_context", "title": "Latent_Space_Autoregressive"},
    {"arxiv_id": "2506.01529", "category": "infinite_context", "title": "Abstract_World_Models"},
    
    # Multimodal Learning
    {"arxiv_id": "2411.06284", "category": "infinite_context", "title": "Multimodal_LLM_Survey"},
    {"arxiv_id": "2503.08497", "category": "infinite_context", "title": "Multi_Modal_Representation"},
    {"arxiv_id": "2503.04839", "category": "infinite_context", "title": "Multimodal_In_Context"},
    {"arxiv_id": "2410.05160", "category": "infinite_context", "title": "VLM2Vec"},
    {"arxiv_id": "2508.19294", "category": "infinite_context", "title": "Object_Detection_VLM"},
    {"arxiv_id": "2403.12736", "category": "infinite_context", "title": "Multimodal_In_Context_Vision"},
    {"arxiv_id": "2510.09586", "category": "infinite_context", "title": "Vision_Language_Survey"},
    {"arxiv_id": "2509.25373", "category": "infinite_context", "title": "Vision_Language_Reasoning"},
    {"arxiv_id": "2404.01322", "category": "infinite_context", "title": "Multi_Modal_Review"},
    
    # Federated Learning
    {"arxiv_id": "2509.10691", "category": "infinite_context", "title": "Privacy_Preserving_Federated"},
    {"arxiv_id": "2511.08998", "category": "infinite_context", "title": "Enterprise_Privacy_Preserving"},
    {"arxiv_id": "2504.17703", "category": "infinite_context", "title": "Federated_Learning_Survey"},
    
    # Neural ODEs
    {"arxiv_id": "2503.03129", "category": "infinite_context", "title": "Neural_ODEs_2025"},
    {"arxiv_id": "2504.10373", "category": "infinite_context", "title": "Neural_ODEs_Applications"},
    {"arxiv_id": "2504.08769", "category": "infinite_context", "title": "High_Order_Neural_ODEs"},
    {"arxiv_id": "2503.23167", "category": "infinite_context", "title": "Graph_ODEs_Survey"},
    {"arxiv_id": "2507.19036", "category": "infinite_context", "title": "Neural_ODEs_Learning"},
    {"arxiv_id": "2507.03860", "category": "infinite_context", "title": "Taylor_Model_PINNs"},
    {"arxiv_id": "2502.15642", "category": "infinite_context", "title": "Training_Neural_ODEs"},
    {"arxiv_id": "2510.09685", "category": "infinite_context", "title": "DNN_Differential_Equations"},
    {"arxiv_id": "2409.20150", "category": "infinite_context", "title": "Deep_Learning_ODEs"},
    {"arxiv_id": "2408.06073", "category": "infinite_context", "title": "Neural_ODEs_Model_Reduction"},
    {"arxiv_id": "2409.16471", "category": "infinite_context", "title": "Score_Based_Neural_ODEs"},
    
    # Self-Supervised Learning
    {"arxiv_id": "2506.04411", "category": "infinite_context", "title": "Self_Supervised_Contrastive"},
    {"arxiv_id": "2510.10572", "category": "infinite_context", "title": "Understanding_Self_Supervised"},
    {"arxiv_id": "2510.08852", "category": "infinite_context", "title": "Supervised_Self_Supervised_Alignment"},
    {"arxiv_id": "2410.14673", "category": "infinite_context", "title": "Contrastive_Nonlinear_System"},
    {"arxiv_id": "2510.25560", "category": "infinite_context", "title": "Controlling_Contrastive"},
    {"arxiv_id": "2409.04607", "category": "infinite_context", "title": "Self_Supervised_Videos"},
    {"arxiv_id": "2409.07402", "category": "infinite_context", "title": "Multimodal_Contrastive_Alignment"},
    {"arxiv_id": "2408.16965", "category": "infinite_context", "title": "Contrastive_Synthetic_Positives"},
    
    # Graph Neural Networks
    {"arxiv_id": "2502.16454", "category": "infinite_context", "title": "GNN_2025"},
    {"arxiv_id": "2502.06784", "category": "infinite_context", "title": "RelGNN"},
    {"arxiv_id": "2509.01170", "category": "infinite_context", "title": "ADMP_GNN"},
    {"arxiv_id": "2412.19419", "category": "infinite_context", "title": "Introduction_to_GNN"},
    {"arxiv_id": "2412.08193", "category": "infinite_context", "title": "MoE_Message_Passing"},
    {"arxiv_id": "2502.06136", "category": "infinite_context", "title": "GNN_Fraction"},
    {"arxiv_id": "2501.18739", "category": "infinite_context", "title": "Beyond_Message_Passing"},
    {"arxiv_id": "2508.19884", "category": "infinite_context", "title": "Parameter_Free_Message_Passing"},
    {"arxiv_id": "2505.23185", "category": "infinite_context", "title": "Effective_Receptive_Field"},
    {"arxiv_id": "2505.00291", "category": "infinite_context", "title": "Recurrent_GNN"},
    {"arxiv_id": "2506.22084", "category": "infinite_context", "title": "Transformers_are_GNN"},
    
    # RLHF
    {"arxiv_id": "2402.09401", "category": "infinite_context", "title": "RLHF_Active_Queries"},
    {"arxiv_id": "2504.12501", "category": "infinite_context", "title": "RLHF_Survey"},
    {"arxiv_id": "2312.14925", "category": "infinite_context", "title": "RLHF_Comprehensive_Survey"},
    
    # Scaling Laws
    {"arxiv_id": "2502.12051", "category": "infinite_context", "title": "Upscale_Scaling_Law"},
    {"arxiv_id": "2405.18392", "category": "infinite_context", "title": "Compute_Optimal_Beyond_Fixed"},
    {"arxiv_id": "2409.17858", "category": "infinite_context", "title": "Feature_Learning_Scaling"},
    {"arxiv_id": "2405.15074", "category": "infinite_context", "title": "4_3_Phases_Scaling"},
    {"arxiv_id": "2001.08361", "category": "infinite_context", "title": "Scaling_Laws_Original"},
    
    # Mixture of Experts
    {"arxiv_id": "2410.07524", "category": "infinite_context", "title": "Upcycling_MoE"},
    {"arxiv_id": "2503.00245", "category": "infinite_context", "title": "CoSMoEs"},
    {"arxiv_id": "2508.18672", "category": "infinite_context", "title": "Optimal_Sparsity_MoE"},
    {"arxiv_id": "2507.11181", "category": "infinite_context", "title": "MoE_in_LLM"},
    {"arxiv_id": "2511.04805", "category": "infinite_context", "title": "PuzzleMoE"},
    {"arxiv_id": "2407.04153", "category": "infinite_context", "title": "Million_Experts"},
    
    # Test-Time Compute
    {"arxiv_id": "2502.12215", "category": "infinite_context", "title": "Test_Time_Scaling_o1"},
    {"arxiv_id": "2501.19393", "category": "infinite_context", "title": "s1_Simple_Test_Time"},
    {"arxiv_id": "2510.14232", "category": "infinite_context", "title": "Test_Time_IOI_Gold"},
    {"arxiv_id": "2501.02497", "category": "infinite_context", "title": "Test_Time_Survey"},
    {"arxiv_id": "2408.03314", "category": "infinite_context", "title": "Scaling_Test_Time_Optimally"},
]

print(f"Starting download of {len(papers_to_download)} papers...")
print("=" * 60)

successful = 0
failed = 0

for paper in papers_to_download:
    if download_arxiv_paper(paper["arxiv_id"], paper["category"], f"{paper['title']}.pdf"):
        successful += 1
    else:
        failed += 1

print("=" * 60)
print(f"\nDownload Summary:")
print(f"✓ Successful: {successful}")
print(f"✗ Failed: {failed}")
print(f"Total: {len(papers_to_download)}")
print(f"\nPapers saved in:")
print(f"  - neuromorphic/")
print(f"  - infinite_context/")