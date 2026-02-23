# 2026 WSAI Summer Internship — Statements of Purpose

---

## PROJECT 001 — Federated Learning
**PI:** Dr. Saurav Prakash | **Interns:** 2 | **Duration:** 6 months

**Statement of Purpose:**

Since McMahan et al. introduced FedAvg in 2017, federated learning has moved from a theoretical curiosity to a practical framework — yet the gap between theory and deployment on real edge hardware remains wide. The problem of statistical heterogeneity across clients (the non-IID data problem) is well-documented, and approaches like FedProx, SCAFFOLD, and FedNova have attempted different corrections. What interests me about this project is the intersection it targets: resource constraints, fairness, and privacy — simultaneously. Most existing work optimises for one at the cost of another.

My coursework in Linear Algebra and Probability provides the mathematical grounding needed for understanding convergence guarantees and gradient aggregation schemes. The ML and DL courses covered optimisation landscapes, loss surfaces, and regularisation — all directly relevant to FL model training. The image classification project I worked on involved distributed data splits for validation, which, at a conceptual level, mirrors the client-server data partitioning in FL. Python is the primary tool, and I have working fluency.

The specific task of developing algorithms addressing memory, computation, and communication constraints on edge devices is something I want to engage with through hands-on experimentation — particularly exploring model compression techniques like pruning and quantisation within the FL pipeline, and studying how differential privacy mechanisms affect convergence rates when fairness constraints are imposed.

---

## PROJECT 002 — Differentially Private Synthetic Data Generation
**PI:** Dr. Krishna Pillutla | **Interns:** 1 | **Duration:** 6 months

**Statement of Purpose:**

Differential privacy, formalised by Dwork et al., provides the strongest known guarantee for individual data protection, but applying it to generative models — especially large language models — remains an active challenge. DP-SGD (Abadi et al., 2016) established the foundational mechanism for private training, but the privacy-utility tradeoff in text generation is sharper than in structured data because of the sequential, high-dimensional nature of language. Recent work on private inference-time approaches and synthetic data using foundation models (e.g., DP-Few-Shot Generation) suggests a shift from training-time to inference-time privacy, which is where this project appears positioned.

My coursework in Probability and Statistics provides the background for understanding privacy budgets (epsilon-delta guarantees) and noise calibration. The ML course covered generalisation theory, which connects directly to understanding how privacy noise affects model utility. From the DL course, I understand transformer architectures and attention mechanisms at a structural level, which is necessary for working with generative language models. I have experience with Python and model training pipelines.

The project requires reading, understanding, and critiquing scientific papers, then implementing and iterating. This aligns with how I approach learning — through implementation rather than passive reading. The mathematical maturity requirement around proofs and optimisation is something my Linear Algebra and Probability coursework has prepared me for, and I am prepared to invest the additional effort for any further background.

---

## PROJECT 003 — Design of Neural Architectures for Structured Data
**PI:** Dr. Lakshmi Narasimhan Theagarajan | **Interns:** 2 | **Duration:** 4 months

**Statement of Purpose:**

The problem of applying deep learning to structured/tabular data has seen growing attention since Arik and Pfister's TabNet (2019) demonstrated that attention-based architectures can outperform gradient-boosted trees on certain benchmarks. More recently, TabTransformer, FT-Transformer, and SAINT have explored how self-attention and feature embeddings can capture inter-column dependencies that standard MLPs miss. Yet XGBoost and LightGBM still dominate many Kaggle-style benchmarks on tabular data, suggesting that neural approaches for structured data remain an open problem.

This project sits squarely in my area of strength. My DL coursework covered CNNs, RNNs, attention mechanisms, and the design principles behind different architectures. The image classification project I implemented involved constructing custom architectures in Python, training with different optimisers (SGD, Adam), evaluating with multiple metrics, and performing hyperparameter tuning — tasks 1 through 5 listed in the project description are essentially what I have already practised. The difference here is the data modality: structured rather than visual.

Graph Theory from my coursework is directly relevant when the structured data involves relational features or linked tables — representing these as graph structures and applying message-passing networks is a natural extension. I want to explore hybrid architectures that combine the inductive biases of tree-based methods with the representational flexibility of neural networks, and benchmark systematically against existing approaches.

---

## PROJECT 004 — Accurate and Efficient LLM-based Materials Data Extraction
**PI:** Dr. Rohit Batra | **Interns:** 1 | **Duration:** 3–6 months

**Statement of Purpose:**

Information extraction from scientific literature has evolved from rule-based NER systems to transformer-based pipelines, but materials science poses unique challenges: domain-specific nomenclature, data scattered across tables, figures, and running text, and the need for cross-referencing between composition, processing, and property data. Recent work like MatSciBERT and Structured Information Inference (SII) pipelines have made progress, but the agentic paradigm — where multiple specialised LLM agents handle different extraction tasks collaboratively — is relatively unexplored in materials informatics.

The two-pipeline structure of this project (extraction + evaluation) is well-conceived. From my ML coursework, I understand supervised pipelines and evaluation methodology — accuracy, precision, recall, F1 — which transfer directly to assessing extraction quality. My DL course introduced transformer architectures, and my Python experience includes building end-to-end training and inference pipelines.

The agentic framework aspect is what draws me here. Designing specialised agents — one for tables, one for figures, one for composition resolution — requires understanding how to decompose a complex task into modular, interacting components. My DSA coursework trained me in this kind of modular problem decomposition. I see this project as an opportunity to work with the full LLM stack: prompt engineering, RAG, and potentially fine-tuning — each layer addressing a different failure mode of the extraction pipeline. The evaluation pipeline, which needs to confirm accuracy with minimal human input, is an interesting meta-problem in itself.

---

## PROJECT 005 — ML Model for Design of Sustainable Polymers for CO2 Capture
**PI:** Dr. Rohit Batra | **Interns:** 1 | **Duration:** 3–6 months

**Statement of Purpose:**

Molecular property prediction using graph neural networks has progressed rapidly since Gilmer et al.'s Message Passing Neural Networks (MPNN, 2017). For polymers specifically, the challenge is representing repeat units and network topologies that differ fundamentally from small molecules. Recent work on polymer informatics — including polyBERT and graph-based representations of porous polymer networks (PPNs) — has shown that combining molecular graph features with learned text representations can improve prediction accuracy, particularly for properties like CO2 adsorption capacity and selectivity where data is scarce.

Graph Theory is part of my coursework, which gives me a structural understanding of graph representations, adjacency matrices, and traversal algorithms. This maps directly to understanding how GNNs operate — message passing over molecular graphs is essentially a learned aggregation over node neighbourhoods. My ML and DL courses covered loss functions, optimisation, and transfer learning concepts. The image classification project I implemented used transfer learning from pre-trained models, so the concept of fine-tuning a foundation model on a downstream task is familiar ground.

The core challenge here — finding the right architecture to combine graph, LLM, and hand-crafted features — is an architecture search and feature fusion problem. I want to explore how attention-based fusion (versus simple concatenation or gating) affects prediction quality, and how transfer learning from larger molecular datasets can compensate for the limited PPN dataset size.

---

## PROJECT 006 — Knowledge Distillation for Disease Burden Estimation
**PI:** Prof. Nirav Bhatt | **Interns:** 3 | **Duration:** 4–6 months

**Statement of Purpose:**

Knowledge distillation, introduced by Hinton et al. (2015), established that a smaller student network can learn to mimic a larger teacher by training on soft label distributions rather than hard targets. The extension to settings with unlabelled data connects to semi-supervised learning frameworks — pseudo-labelling, consistency regularisation (FixMatch, MixMatch), and more recently, self-distillation approaches where the teacher and student share architectures. When the data is epidemiological — sparse, noisy, and with weak biological signals — the theoretical analysis of when and why distillation helps becomes critical.

This project demands strong mathematical foundations, which is exactly where my preparation lies. Linear Algebra, Probability, and the optimisation components of my ML course provide the tools for formulating and analysing the mathematical optimisation problems described. The project mentions theoretical analysis and algorithm development grounded in theory — this is distinct from pure implementation work, and requires the ability to reason about convergence, generalisation bounds, and label noise.

My experience with model training and systematic evaluation from the image classification project translates here: understanding training dynamics, diagnosing underfitting versus overfitting, and designing proper validation schemes. The epidemiological application provides concrete grounding for otherwise abstract mathematical frameworks. I am interested in exploring how the bias-variance characteristics of distilled models change under different label noise regimes and data sparsity levels.

---

## PROJECT 007 — Generative AI for Fragment-based Lead Optimisation in Drug Design
**PI:** Dr. Agastya P Bhati | **Interns:** 2 | **Duration:** 3–6 months

**Statement of Purpose:**

Fragment-based drug design has traditionally relied on medicinal chemists' intuition for generating congeneric series, but reinforcement learning offers a systematic alternative. REINVENT (Olivecrona et al., 2017) demonstrated that RL can guide a generative model toward molecules with desired properties, and subsequent work (REINVENT 2.0, 3.0) has added multi-objective optimisation and curriculum learning. The key gap this project addresses is the systematic coupling of RL-based structure generation with physics-based relative binding free energy (RBFE) methods — using simulation accuracy as the reward signal rather than cheaper QSAR proxies.

My DL coursework covered the architectural foundations relevant here: sequence generation, reward-based training, and policy gradient methods at a conceptual level. The image classification project involved iterative training and evaluation workflows, which parallels the iterative cycle of generate-simulate-evaluate that this project proposes. Python and PyTorch, the primary tools for building the RL model, are within my working toolkit.

The project description notes that the RL model may need to be written from scratch depending on the use-case — this kind of ground-up implementation is where real learning happens. While the MD simulation component involves domain knowledge in computational chemistry, the ML component (developing the RL model, defining reward functions, training the generator) is well within the scope of my current preparation. I am prepared to learn the simulation aspects as needed — the PI's group provides training in these methods, which is one of the stated learning outcomes.

---

## PROJECT 008 — Efficient Language Model Training via Reinforcement Learning Feedback
**PI:** Prof. Nandan Sudarsanam | **Interns:** 2 | **Duration:** 6 months

**Statement of Purpose:**

The alignment of language models through reinforcement learning has become the central post-training paradigm since InstructGPT (Ouyang et al., 2022). PPO remains the workhorse, but its instability and computational overhead have motivated alternatives: DPO (Rafailov et al., 2023) eliminates the reward model entirely, while GRPO (Shao et al., 2024), used in DeepSeek-R1, replaces the critic with group-relative advantages. Offline RL approaches like IQL and CQL offer another axis — training on static preference datasets without online generation. Each paradigm makes different tradeoffs between sample efficiency, stability, and compute.

My ML coursework covered the optimisation foundations (gradient descent, policy gradient theorem at a high level) and my DL course covered transformer architectures, which are the models being aligned. Understanding how PPO's clipped objective prevents catastrophic policy updates, or how GRPO computes advantages without a value function, requires exactly the kind of mathematical reasoning my Linear Algebra and Probability courses developed.

The project focuses on preliminary survey and basic implementations, which suggests a research-oriented structure: read, understand, implement, compare. This is the workflow I follow in my own learning. The end goal of training context-enriched SLMs (smaller language models) is practically grounded — making alignment techniques work at small scale is arguably harder than at large scale because there is less capacity to absorb the RL signal. I want to understand these tradeoffs empirically.

---

## PROJECT 009 — Accelerating Optimisation using AI in Networks
**PI:** Prof. Sridharakumar Narasimhan | **Interns:** 2 | **Duration:** 2–6 months

**Statement of Purpose:**

Using machine learning to accelerate combinatorial optimisation has seen significant progress since Khalil et al. (2017) showed that GNNs can learn branching policies for MILP solvers. More recent work — such as Gasse et al.'s Ecole framework and Cappart et al.'s survey on ML for combinatorial optimisation — has demonstrated that ML can reduce solve times by predicting good initial solutions, pruning search spaces, or learning decomposition strategies. For network optimisation specifically (transport scheduling, water grid management), the graph structure of the problem maps naturally onto GNN architectures.

This project is a strong fit for my preparation. Graph Theory is part of my coursework — I understand graph representations, spectral properties, and algorithmic approaches to graph problems. My ML course covered supervised and unsupervised learning on structured data, and the DL course introduced architecture design principles. DSA provides the algorithmic thinking needed for understanding solver internals (branch and bound, cutting planes).

The task of integrating trained ML models with standard solvers (Gurobi, CPLEX, HiGHS) to create an end-to-end acceleration pipeline is an engineering challenge that requires both ML competence and systems thinking. My OS and Computer Networks coursework provides familiarity with systems-level reasoning. I am particularly interested in the GNN component: learning to predict which constraints are likely active at optimality, thereby reducing the effective problem size before the solver begins.

---

## PROJECT 010 — Evaluation of LLM for Agentic Tasks and Deployability
**PI:** Prof. Sridharakumar Narasimhan | **Interns:** 2 | **Duration:** 3–6 months

**Statement of Purpose:**

The deployment of LLMs as autonomous agents — capable of planning, tool use, and multi-step reasoning — on edge hardware represents a convergence of two active research areas: agentic AI (ReAct, Toolformer, HuggingGPT) and efficient inference (quantisation via GPTQ/AWQ, pruning, knowledge distillation for deployment). The central tension is clear: agentic behaviours like chain-of-thought reasoning and self-correction require high model capacity, while edge deployment demands aggressive compression. How much reasoning survives quantisation is an empirical question that needs systematic study.

My DL coursework covered the architectural foundations of the models being evaluated — transformer attention, layer normalisation, positional encoding. The image classification project gave me practical experience in evaluating model performance under different configurations, measuring accuracy against compute cost — the same analytical framework needed here, applied to latency, memory, and reasoning accuracy. My OS coursework provides understanding of memory management and process scheduling relevant to edge deployment.

The benchmarking and evaluation methodology is what I find most interesting. Defining what "agentic capability" means at different quantisation levels, designing experiments that isolate reasoning degradation from perplexity changes, and measuring real-world inference latency on constrained hardware — this is measurement science applied to AI, and it requires careful experimental design rather than just running models.

---

## PROJECT 011 — Deployment of Open Source IoT Stack with ML for Analytics
**PI:** Prof. Sridharakumar Narasimhan | **Interns:** 3 | **Duration:** 3–6 months

**Statement of Purpose:**

The integration of IoT device networks with local ML analytics — bypassing cloud dependency — addresses both latency and data sovereignty concerns. Protocol selection between MQTT, CoAP, HTTP, TCP, and UDP involves well-studied tradeoffs: MQTT's publish-subscribe model suits event-driven sensor data, CoAP provides RESTful semantics for constrained devices, and the choice affects throughput, reliability, and power consumption in measurable ways.

My Computer Networks coursework provides direct preparation for understanding protocol stacks, packet structures, and the tradeoffs between connection-oriented and connectionless communication. Operating Systems coursework covers process management, memory allocation, and system calls — all relevant to hosting and managing a local server. Python, the primary development language, is my working tool, and the ML component (analytics on collected sensor data) connects to my ML coursework.

The project involves embedded C programming for devices, server hosting, IP mapping, and tunnelling — a full-stack IoT deployment. My interest here is in the data analytics layer: applying ML models to time-series sensor data in real-time on the self-hosted server, optimising for the latency constraints imposed by the protocol choice. The systems integration aspect — making sensors, protocols, server, and ML pipeline work together — is the kind of end-to-end engineering challenge that builds practical competence.

---

## PROJECT 012 — Internet of Things for Edge Devices using Machine Learning
**PI:** Prof. Sridharakumar Narasimhan | **Interns:** 3 | **Duration:** 3–6 months

**Statement of Purpose:**

Edge analytics on resource-constrained IoT devices requires rethinking standard ML pipelines. TinyML — running inference on microcontrollers with kilobytes of RAM — has been enabled by frameworks like TensorFlow Lite Micro and tools for model quantisation and pruning. The challenge extends beyond model size: time-series analysis, spectral decomposition, and image processing must be implemented efficiently using libraries like NumPy and SciPy within the power and memory budgets of devices like ESP32 or Raspberry Pi Pico.

My coursework in ML and DL provides the algorithmic foundation for designing models suitable for edge deployment — understanding which architectures can be compressed without significant accuracy loss. The image classification project involved model training and evaluation in Python, and the architectural awareness gained (layer sizes, parameter counts, computational cost per layer) is directly relevant to selecting models that fit edge constraints. OS coursework covers the systems-level concerns: memory management, scheduling, and resource allocation.

The project emphasises efficient data transmission using lightweight protocols and power-optimised frameworks like MicroPython. I am interested in the specific challenge of selective metric transmission — deciding which processed features to send upstream versus which to act on locally. This is fundamentally a compression and information theory problem applied to sensor data, and optimising it requires understanding both the ML model's requirements and the communication channel's constraints.

---

## PROJECT 013 — Application of Deep Learning to Water Distribution Networks
**PI:** Prof. Sridharakumar Narasimhan | **Interns:** 3 | **Duration:** 2–6 months

**Statement of Purpose:**

GANs for anomaly detection in infrastructure networks is a well-motivated application. The standard approach — training a GAN on normal operating data and using reconstruction error or discriminator score as an anomaly signal — has been applied in manufacturing (AnoGAN, Schlegl et al., 2017) and network intrusion detection. Applying this to water distribution networks for contaminant entry detection and leak localisation adds spatial structure: the network topology constrains how anomalies propagate, making graph-based or spatially-aware GAN architectures a natural choice.

My DL coursework covered generative models including GANs — generator-discriminator dynamics, mode collapse, training stability — and the image classification project gave me practical experience with training deep networks and diagnosing training pathologies. The project description mentions three specific applications: contaminant detection, leak localisation, and network reconstruction from road maps. Each requires a different modelling approach, and the variety is appealing.

Graph Theory from my coursework is relevant for representing WDN topology — nodes as junctions, edges as pipes — and understanding how information (or contaminant) flows through the network. The reconstruction task (inferring pipe network layout from road maps) has a computer vision flavour that connects to my interest in image-based deep learning. I want to explore how combining graph structure with spatial features improves detection and localisation accuracy over purely data-driven approaches.

---

## PROJECT 014 — Language Structure, Modality, and Cognitive Load: An Eye-tracking Study
**PI:** Dr. Anindita Sahoo | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Eye-tracking as a window into cognitive processing during reading has a rich psycholinguistic tradition — from Rayner's (1998) comprehensive review of eye movements in reading to recent work on fixation-related metrics (first fixation duration, gaze duration, regression probability) as indices of processing difficulty. The proposed comparison across Indo-Aryan and Dravidian languages is particularly interesting because morphological complexity and orthographic transparency differ systematically between these language families, providing a natural experiment.

While my primary training is in computer science and machine learning, several aspects of this project align with my skills. The data analysis component — processing fixation duration data, computing regression frequencies, statistical testing across conditions — requires exactly the statistical and programming skills from my Probability & Statistics coursework and Python experience. The project description mentions that AI methods will help predict which sentence structures increase processing difficulty — this is a supervised learning problem where eye-tracking features are inputs and processing difficulty is the target.

The stimuli design aspect (visual and lexical) involves systematic experimental methodology, which my coursework in ML has introduced through concepts like controlled experiments, confounding variables, and proper evaluation design. The cross-linguistic nature of the study — comparing reading patterns across scripts with different visual structures — creates a rich dataset for ML-based analysis of reading behaviour.

---

## PROJECT 015 — Agentic Framework for Gene-Disease Causal Relation Extraction
**PI:** Prof. Manikandan Narayanan | **Interns:** 1 | **Duration:** 3–4 months

**Statement of Purpose:**

Biomedical relation extraction has progressed from pattern-based approaches through BioBERT and PubMedBERT to agentic systems that orchestrate multiple specialised models. The distinction between causal relations and mere associations in gene-disease literature is subtle — requiring understanding of experimental context (RNAi knockdown vs. observational GWAS) that simple NER + relation classification pipelines miss. The CRED dataset provides structured supervision for this distinction, but scaling beyond it requires the kind of multi-agent reasoning this project proposes.

My DL coursework covered transformer architectures in detail — self-attention, positional encoding, fine-tuning strategies — which directly applies to the LLM fine-tuning task (Llama-3, BioBERT on CRED). The concept of LoRA and parameter-efficient tuning, while not part of my formal coursework, is something I have studied independently given its centrality to current DL practice. My experience with model training workflows (from the image classification project) provides the practical foundation for managing training runs, evaluating outputs, and iterating.

The agentic architecture — where separate agents handle entity recognition, experimental data cross-referencing, and prediction extraction — is a system design problem. My DSA coursework trains this kind of modular thinking: defining interfaces between components, managing data flow, handling edge cases. Building the end-to-end pipeline from raw PubMed text to structured Gene → Disease causal outputs is the kind of engineering-meets-research work that builds genuine capability.

---

## PROJECT 016 — Bias-Variance Analysis of ML/DL Models Predicting Disease Risk
**PI:** Prof. Manikandan Narayanan | **Interns:** 1 | **Duration:** 3 months

**Statement of Purpose:**

Polygenic risk scores (PRS) — linear combinations of SNP effects estimated from GWAS — are the dominant approach for genetic disease risk prediction, but they are known to overfit in high-dimensional settings where features (SNPs) vastly outnumber samples. Regularisation approaches (Ridge, Lasso, Elastic Net) mitigate this by controlling model complexity, but the bias-variance tradeoff manifests differently in genomic data because of linkage disequilibrium (correlation between nearby SNPs) and population stratification. Whether non-linear models (random forests, neural networks) can improve generalisation in this setting is an open question with mixed evidence.

This project maps directly onto my ML coursework, where regularisation, bias-variance decomposition, cross-validation, and model selection were core topics. The image classification project provided hands-on experience with overfitting diagnostics — monitoring training versus validation loss, experimenting with dropout and weight decay, evaluating generalisation on held-out data. These are the same skills needed here, applied to a different data modality.

The project's dual goal — practical guidelines for reliable disease risk models AND highlighting how classical ML concepts apply in real-world scientific settings — resonates with my understanding of ML as a principled methodology rather than just a toolkit. I am interested in systematically studying how different regularisation strategies (L1, L2, group lasso for LD-aware penalisation) affect generalisation across simulated datasets with varying genetic architectures.

---

## PROJECT 017 — Data Fusion for Driver Monitoring
**PI:** Prof. Lelitha Devi Vanajakshi | **Interns:** 1 | **Duration:** 3 months

**Statement of Purpose:**

Multimodal sensor fusion for driver behaviour analysis typically combines inertial measurement unit (IMU) data (accelerometer, gyroscope) with video-based features (head pose, gaze direction, lane position). The fusion can happen at the data level (early fusion), feature level, or decision level — each with different computational and accuracy tradeoffs. IMU-based driving style classification has been studied extensively using time-series features (jerk, angular velocity statistics) fed into classifiers like SVMs, random forests, and LSTMs.

This project is concise and well-defined: fuse IMU and video data, classify driving style. My ML coursework covered classification algorithms — SVMs, decision trees, ensemble methods, neural classifiers — and the evaluation methodology (confusion matrices, precision-recall, cross-validation) needed to assess them. The image classification project provides experience with the visual modality. Python and data analysis tools (pandas, scikit-learn, potentially PyTorch for deep fusion) are the working tools.

The fusion aspect is what interests me most. Deciding how to combine temporally aligned but fundamentally different data streams — continuous IMU signals versus discrete video frames — requires thoughtful feature engineering or an architecture that learns the alignment. I want to explore whether attention-based fusion, where the model learns to weight IMU versus visual cues differently in different driving contexts, improves over simpler concatenation approaches.

---

## PROJECT 018 — Discerning Ecological Behaviour of Minimal Reactomes of the Human Gut
**PI:** Prof. Karthik Raman | **Interns:** 1 | **Duration:** 3–4 months

**Statement of Purpose:**

Constraint-based metabolic modelling, particularly flux balance analysis (FBA), provides a principled way to study microbial metabolism without kinetic parameters — using stoichiometric constraints and linear programming to predict growth rates and metabolic fluxes. The AGORA database contains genome-scale metabolic models for hundreds of gut microbes, and the MinReact algorithm identifies minimal reactomes — the smallest set of reactions supporting growth. Whether metabolic minimisation leads to better ecological cooperation or intensified competition is a question that connects metabolic modelling to ecological theory (Black Queen Hypothesis).

My Linear Algebra coursework is directly relevant — FBA is fundamentally a linear programming problem operating on stoichiometric matrices, and understanding null spaces, rank, and feasibility conditions is essential. Probability and Statistics support the analysis of metabolic flux distributions and secretome comparisons across conditions. Python is one of the required skills, and my DSA background is relevant for algorithm engineering (extending MinReact, Task 6).

The project involves computational modelling, data analysis, and algorithm development — all skills I can contribute to. The biological reasoning component (interpreting flux distributions, understanding metabolic pathway significance) is something I am prepared to develop through reading and mentorship. The computational problems are clearly defined and mathematically grounded, which suits my analytical approach.

---

## PROJECT 019 — Visual SLAM Mapping and Evaluation on a Scaled Autonomous Traffic Testbed
**PI:** Prof. Ramkrishna Pasumarthy / Prof. Nirav Bhatt | **Interns:** 1 | **Duration:** 4 months

**Statement of Purpose:**

Visual SLAM has evolved from MonoSLAM through ORB-SLAM to ORB-SLAM3, which unifies monocular, stereo, and visual-inertial modes under a single framework. The key challenges for deployment on scaled autonomous vehicles — limited compute on platforms like Jetson Orin Nano, motion blur from rapid manoeuvres, and varying lighting on indoor testbeds — require systematic benchmarking rather than just running the algorithm out of the box. Parameter tuning (feature extraction thresholds, keyframe insertion policy, loop closure sensitivity) can dramatically affect both accuracy and real-time feasibility.

Computer Vision is one of my core interests, and SLAM sits at the intersection of vision and geometry. My DL coursework covered feature extraction concepts (convolutional filters, learned representations), and my Linear Algebra background provides the mathematical foundation for understanding pose estimation (rotation matrices, homogeneous transformations, epipolar geometry). The image classification project involved working with visual data pipelines in Python.

The benchmarking and evaluation aspect of this project is well-structured: setup ORB-SLAM3, generate maps, compare with alternatives, analyse errors using standard metrics (ATE, RPE), and optimise for real-time execution. This systematic evaluation methodology mirrors how I approach ML experiments — controlled comparisons, quantitative metrics, and parameter sensitivity analysis. Working with real sensor data on physical hardware adds a dimension that pure simulation work cannot replicate.

---

## PROJECT 020 — Learning-Based Control Design for Scaled Autonomous Vehicles
**PI:** Prof. Ramkrishna Pasumarthy / Prof. Nirav Bhatt | **Interns:** 1 | **Duration:** 4 months

**Statement of Purpose:**

Reinforcement learning for vehicle control has been demonstrated in simulation (CARLA, TORCS) but deploying RL controllers on physical hardware introduces challenges that simulation-trained policies often fail to handle: sensor noise, actuator delays, and the sim-to-real gap. The comparison between RL-based and classical controllers (PID, LQR) is valuable because it quantifies what learning adds over well-tuned baselines — and the answer is not always clear for simple control tasks on constrained hardware.

My ML coursework introduced the foundations of reward-based learning, and my DL course covered the function approximation side (neural networks as policy or value function approximators). Linear Algebra provides the mathematical tools for understanding state-space representations and controller analysis. The image classification project's training loop — forward pass, loss computation, backpropagation, evaluation — maps onto the RL training loop: action selection, reward collection, policy update, evaluation.

The project tasks are well-sequenced: implement PID baselines, design RL controllers, train in simulation, deploy on testbed, compare. This progression from classical to learned approaches, from simulation to hardware, builds understanding at each stage. I am particularly interested in analysing convergence behaviour and safety — how to ensure the RL controller does not produce unsafe actions during training on the physical vehicle, potentially using constrained policy optimisation or action clipping.

---

## PROJECT 021 — Leveraging GenAI and LLMs for Robust Design, Verification, and Testing of Industrial Control Panels
**PI:** Prof. Satyanarayan Seshadri | **Interns:** 1 | **Duration:** 6 months

**Statement of Purpose:**

Automating the interpretation of engineering documentation — particularly P&ID diagrams — has been attempted through symbol detection (YOLO, Faster R-CNN), OCR for annotations, and rule-based logic extraction. The challenge is that P&IDs encode both spatial relationships (pipe connections, instrument placement) and logical relationships (control sequences, interlocks) that require multi-modal understanding. Recent advances in document AI (LayoutLM, Donut) and visual question answering suggest that transformer-based models can bridge visual and textual modalities, but applying them to engineering documentation at industrial scale is largely unexplored.

My Computer Vision interest and DL coursework align directly with the core technical challenge: extracting structured information from visual engineering documents. The image classification project involved processing visual inputs through convolutional architectures, and extending this to object detection (symbols, instruments in P&IDs) is a natural progression. My understanding of neural architectures covers both the vision backbone (CNN/ViT for feature extraction) and the language components (transformers for understanding annotations and specifications).

The project mentions graph-based reasoning for system representation — representing control dependencies as directed graphs and validating operational sequences — which connects to my Graph Theory coursework. The additional note that top performers may be absorbed full-time indicates the project's industrial relevance and the PI's investment in finding committed contributors. The 6-month duration allows for deep engagement with both the AI methodology and the engineering domain.

---

## PROJECT 022 — Climate Action Tool
**PI:** Prof. Satyanarayan Seshadri | **Interns:** 2 | **Duration:** 4 months

**Statement of Purpose:**

Integrated assessment models (IAMs) for climate action — such as DICE, REMIND, and MESSAGE — have traditionally been the domain of specialised modelling teams. Making these accessible through interactive desktop applications democratises climate scenario analysis. The technical stack described (PySide6/Qt with Python backend, AMPL optimisation solver integration, geospatial visualisation) is essentially a full-stack scientific application development project.

My Database Systems coursework provides the foundation for designing data ingestion pipelines from multiple sources using SQL — one of the two primary intern tasks. Understanding relational schemas, query optimisation, and data normalisation applies directly to structuring climate datasets (emissions inventories, energy system data, decarbonisation pathway parameters) for efficient retrieval and analysis. Python is my working language, and my experience building ML pipelines — which involve similar data preprocessing, transformation, and output generation workflows — transfers to building the data backend.

The systems modelling component involves understanding how interdependencies between sectors (energy, transport, industry) can be represented computationally and solved using optimisation. My ML coursework covered optimisation at the algorithmic level, and extending this to constrained optimisation for scenario analysis is a natural step. The geospatial visualisation component is newer territory, but the combination of data pipeline engineering and scientific computing is where my skills apply most directly.

---

## PROJECT 023 — Pangenome-Driven Population Genomics of Indian Populations
**PI:** Prof. Himanshu Sinha + 3 Co-PIs | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Pangenome references represent a paradigm shift from the single linear reference genome (GRCh38) that has dominated human genomics. The vg toolkit (Garrison et al., 2018) enables variant calling against genome graphs, capturing structural variants that linear alignment misses. For population genetics, graph-based genotyping across 3,000 Indian genomes from GenomeIndia is ambitious — most pangenome studies to date have focused on smaller, less diverse cohorts (HPRC's 47 diploid assemblies).

The computational methods involved — PCA, spectral clustering, PBWT-based phasing — are fundamentally statistical and algorithmic. My Probability & Statistics coursework covers the mathematical foundations of PCA (eigendecomposition of covariance matrices), and my ML course introduced dimensionality reduction and clustering algorithms. Graph Theory is relevant for understanding the pangenome graph structure itself — how haplotype paths traverse variant bubbles.

The bioinformatics-specific tools (vg giraffe, PanGenie, PBWT) require learning new software ecosystems, but the underlying operations — graph traversal, matrix decomposition, hidden Markov models (Li-Stephens) — are computational methods I have encountered in my coursework. Python proficiency, listed as the primary skill requirement, is well within my preparation. The population structure analysis tasks (SV-based PCA, spectral clustering, validation against ethnicity labels) are structured as a systematic computational study, which suits my analytical approach.

---

## PROJECT 024 — Automatic Vehicle Trajectory Extraction from Drone Videos
**PI:** Prof. Bhargava Rama Chilukuri | **Interns:** 2 | **Duration:** 2–3 months

**Statement of Purpose:**

Vehicle detection and tracking from aerial imagery has been addressed using YOLO variants for detection and DeepSORT or ByteTrack for multi-object tracking. The challenge at intersections and roundabouts is specific: occlusion between vehicles, complex turning trajectories, and the need to maintain consistent IDs across multiple camera angles. Recent work on transformer-based trackers (TrackFormer, MOTR) offers end-to-end detection-tracking, but the computational overhead may conflict with the need for processing long drone video sequences.

Computer Vision is a core interest of mine, and this project is directly aligned. My image classification project involved processing visual inputs through deep learning pipelines — detection and tracking extend this to spatial localisation and temporal consistency. The DL coursework covered convolutional architectures that form the backbone of modern detectors. Python is the development language, and OpenCV (for video processing) and PyTorch (for model inference/training) are tools I work with.

The GUI development aspect adds a software engineering dimension to the CV research. Building an interface for trajectory visualisation, annotation correction, and multi-direction video alignment requires thinking about usability alongside algorithm accuracy. The possibility of developing this into open-source software is appealing — it means the code quality, documentation, and interface design matter as much as the detection accuracy, and that combination of engineering rigour with CV research is what I want to develop.

---

## PROJECT 025 — Pedestrian Trajectory Extraction from Dense Crowds
**PI:** Prof. Bhargava Rama Chilukuri | **Interns:** 2 | **Duration:** 2–3 months

**Statement of Purpose:**

Pedestrian tracking in dense crowds is substantially harder than vehicle tracking — smaller targets, more severe occlusion, non-rigid motion, and unpredictable directional changes. The Social Force Model (Helbing and Molnár, 1995) established the physics-based framework for pedestrian dynamics, but extracting trajectories from video requires overcoming the detection-level challenges first. CrowdHuman and MOT benchmarks have pushed detector and tracker performance, but most are evaluated on Western street scenes — structured queues and unstructured gatherings in Indian contexts present different density profiles and movement patterns.

This project extends my CV interest to one of its harder sub-problems. The image classification project taught me about feature representations at the single-image level; tracking requires extending this to temporal sequences with identity association. My DL coursework covered sequence modelling (RNNs, attention over temporal features) that applies to trajectory prediction and tracking consistency. The multi-directional flow aspect — pedestrians moving in crossing directions simultaneously — introduces the data association challenge that makes multi-object tracking non-trivial.

The practical skills required (computer vision course, Python programming) match my preparation. The specific challenge of handling both structured queues and unstructured gatherings within one algorithm — adapting to very different crowd topologies — is an interesting design problem that goes beyond applying off-the-shelf trackers. I want to explore how crowd density estimation can inform tracker parameters dynamically: tighter association thresholds in dense regions, looser in sparse.

---

## PROJECT 026 — Applications using Single Board Computer
**PI:** Prof. Bhargava Rama Chilukuri | **Interns:** 2 | **Duration:** 2–3 months

**Statement of Purpose:**

Deploying computer vision models on single board computers (SBCs) like Raspberry Pi or NVIDIA Jetson Nano bridges the gap between research prototypes and field-deployable traffic monitoring systems. The constraints are concrete: limited GPU memory restricts model size, power budgets limit continuous inference, and thermal throttling can degrade performance over time. Model optimisation techniques — TensorRT conversion, INT8 quantisation, knowledge distillation into smaller architectures — become essential rather than optional.

My CV interest and DL background align with the algorithm development side, while OS coursework provides the systems-level understanding needed for SBC programming — process management, memory constraints, I/O handling. The image classification project involved working within specific computational constraints (GPU memory, batch sizes), which at a smaller scale is analogous to the SBC deployment challenge. Python is the primary tool, though C/C++ for performance-critical components is something I have exposure to through my coursework.

The traffic application context — vehicle counting, speed estimation, classification from roadside cameras — translates CV research into measurable public utility. The note about a possible patent suggests the work aims for novelty in either the algorithmic approach or the deployment methodology. I am interested in exploring how lightweight architectures (MobileNet, EfficientNet-Lite) perform on traffic-specific tasks when deployed on SBC hardware, and whether task-specific pruning can improve the accuracy-latency tradeoff beyond generic compression.

---

## PROJECT 027 — AI Incident Reporting Framework
**PI:** Prof. B. Ravindran / Dr. Geetha Raju | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

AI incident databases like AIID (McGregor, 2021) and OECD's AI Incidents Monitor have established the practice of systematic incident cataloguing, but their taxonomies reflect Western deployment contexts. India's AI ecosystem — welfare distribution algorithms, credit scoring for underbanked populations, deepfake proliferation in regional languages — generates harm categories that existing frameworks underrepresent. CeRAI's discussion paper on a federated national database with a multi-dimensional harm taxonomy is a necessary corrective.

My ML and DL coursework gives me practical understanding of where AI systems fail — overfitting, distribution shift, adversarial vulnerabilities, bias amplification from training data — which provides the technical grounding needed to classify and analyse AI incidents meaningfully. Understanding the technical mechanisms behind failures is essential for developing a "Quantifiable AI Harm Severity Assessment Methodology," which requires mapping technical failure modes to societal impact categories.

The project is research and policy-oriented rather than implementation-heavy, but the AI knowledge requirement is marked as mandatory precisely because the classification frameworks must be technically informed. My understanding of model architectures, training pipelines, and deployment considerations provides the vocabulary needed to engage with domain-specific incident analysis. The literature survey component — reviewing global reporting mechanisms and policies — requires the same systematic review skills I apply when surveying ML approaches for a problem.

---

## PROJECT 028 — Responsible Data Governance Frameworks for AI Systems
**PI:** Prof. B. Ravindran / Dr. Geetha Raju | **Interns:** 2 | **Duration:** 4–6 months

**Statement of Purpose:**

Data governance for AI sits at the intersection of technical implementation and regulatory compliance. India's regulatory landscape — the DPDP Act (2023), Copyright Act, IT Act — creates overlapping obligations for organisations deploying AI, and the fragmentation between these acts creates compliance gaps. The technical dimensions (data lineage tracking, consent management for training data, content safety for generative outputs) require solutions informed by both policy requirements and system architecture.

My understanding of AI systems — from data preprocessing through model training to deployment — provides the technical perspective needed for translating policy requirements into functional requirements (Task 4 in the project description). The DBMS coursework covers data management principles that underpin governance practices: access control, integrity constraints, audit trails, and schema design for tracking data provenance. Understanding how AI models consume and transform data at a technical level is essential for identifying governance gaps.

The retrospective analysis of data-related harms in AI systems (Task 3) requires the ability to trace a deployed system's failure back to its data origins — poisoned training sets, demographic imbalances, inadequate anonymisation. My ML coursework provides the framework for understanding these failure modes technically. The content safety challenge for generative AI — deepfakes, misinformation in Indian languages — connects to my understanding of generative model architectures and their vulnerabilities.

---

## PROJECT 029 — Multilingual Indic Text Embedding Model using Contrastive Learning
**PI:** Dr. Sudarsun Santhiappan | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Cross-lingual sentence embeddings have evolved from early approaches (LASER, Artetxe and Schwenk, 2019) through sentence-BERT (Reimers and Gurevych, 2019) to current contrastive learning frameworks. The InfoNCE loss (van den Oord et al., 2018) — maximising agreement between positive pairs (translations) while pushing apart negatives — has become the standard training objective. For Indian languages, the challenge is that most multilingual models (mBERT, XLM-R) are undertrained on South Indian scripts due to their underrepresentation in training corpora, leading to fragmented tokenisation and poor semantic alignment.

My DL coursework covered transformer architectures in detail — multi-head attention, layer normalisation, pre-training objectives (MLM, NSP) — which are the building blocks of the embedding model. The training methodology (contrastive learning with InfoNCE loss) connects to my understanding of loss function design and optimisation from the ML course. Linear Algebra provides the mathematical foundation: cosine similarity in high-dimensional vector spaces, the geometry of embedding spaces, and how contrastive objectives shape these spaces.

The project involves training a model from scratch rather than fine-tuning, which requires deeper architectural understanding — choosing hidden dimensions, number of layers, attention heads, and vocabulary size. My image classification project involved similar design decisions (architecture selection, hyperparameter tuning) at a smaller scale. The evaluation on cross-lingual retrieval and semantic textual similarity benchmarks requires designing rigorous comparison experiments, which aligns with my systematic approach to model evaluation.

---

## PROJECT 030 — Grapheme-Aware Indic Tokenizer Development
**PI:** Dr. Sudarsun Santhiappan | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Standard subword tokenisation algorithms (BPE, WordPiece, Unigram) treat text as byte or character sequences without awareness of script-specific structure. For Indic scripts (Abugidas), this means a consonant cluster like "क्ष" can be split into its constituent code points, destroying the grapheme cluster that represents a single linguistic unit. This fragmentation increases sequence length (higher inference cost) and degrades model performance because the model must re-learn character composition that the writing system already encodes. The indic-tokenizer project addresses this by treating grapheme clusters as atomic units.

My DSA coursework provides the algorithmic thinking needed for text processing — string algorithms, finite automata for pattern matching, and the analysis of algorithmic efficiency. Python proficiency is essential for both the refactoring work and the benchmarking suite. The project requires understanding Unicode internals (NFC/NFD normalisation forms, grapheme cluster boundaries defined by UAX #29), which is a precise specification-driven domain where attention to detail matters.

The benchmarking aspect — comparing tokenisation efficiency (compression ratio, fertility rate, vocabulary utilisation) across 10+ Indic languages against standards like cl100k_base and SentencePiece — requires designing metrics and running systematic comparisons. This connects to my experience with model evaluation methodology from the ML course. The integration test (plugging the tokenizer into a language model training loop) verifies that improved tokenisation translates to improved downstream performance, closing the loop between component-level and system-level evaluation.

---

## PROJECT 031 — Modular LoRA Adapter Merging for LLMs
**PI:** Dr. Sudarsun Santhiappan | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

LoRA (Hu et al., 2021) made parameter-efficient fine-tuning practical by decomposing weight updates into low-rank matrices, reducing trainable parameters by orders of magnitude. The problem this project addresses — merging multiple adapters without catastrophic interference — connects to a broader line of work: "Model Soups" (Wortsman et al., 2022) showed that averaging fine-tuned models improves robustness, TIES-Merging (Yadav et al., 2023) resolves conflicts through trimming and sign election, and SLERP provides a geometrically principled interpolation on the weight manifold. The fundamental question is whether independently trained skills (coding, Hindi, medical) can be composed in weight space without retraining.

My Linear Algebra coursework is directly relevant — LoRA's core operation is low-rank matrix decomposition (SVD), and understanding how rank, spectral properties, and subspace alignment affect merging outcomes requires solid matrix theory. The DL course covered neural network weight landscapes, loss surface geometry, and the role of initialisation — all pertinent to understanding why simple weight averaging sometimes works and sometimes catastrophically fails.

The experimental methodology — train distinct adapters, apply merging techniques, evaluate for regression on individual tasks — requires the systematic evaluation approach I developed through my image classification project. The "Skill Composition" framing is what makes this intellectually interesting: it suggests that knowledge in neural networks can be modularly composed, challenging the assumption that multi-task learning requires joint training.

---

## PROJECT 032 — Semantic Text Reconstruction from Embeddings
**PI:** Dr. Sudarsun Santhiappan | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Embedding inversion — reconstructing input text from its dense vector representation — probes a fundamental question about information preservation in learned representations. Morris et al. (2023) demonstrated that sentence embeddings can be inverted with surprising fidelity using seq2seq models conditioned on embedding vectors, raising both privacy concerns (embeddings are not as "one-way" as assumed) and opportunities (data augmentation, lossy compression). The quality of reconstruction depends on the embedding model's information bottleneck: how much semantic detail survives the compression into a fixed-dimensional vector.

My DL coursework covered encoder-decoder architectures, the attention mechanism, and how information flows through bottleneck layers — directly relevant to understanding what reconstruction is possible from different embedding models. The concept of conditional generation (generating output conditioned on a specific input representation) connects to my understanding of how decoders in seq2seq models use context vectors. The image classification project involved understanding feature representations at intermediate layers, which is conceptually similar to understanding what information embedding vectors encode.

The evaluation methodology is non-standard: semantic similarity metrics (BERTScore, Mauve) rather than exact-match metrics (BLEU) — because the reconstruction should preserve meaning, not surface form. This distinction between semantic and syntactic evaluation connects to deeper questions about what neural representations capture. The privacy implications (can you reconstruct training data from published embedding models?) add practical urgency to what might otherwise be a purely academic exercise.

---

## PROJECT 033 — Computational Analysis and Synthesis of Tamil Venba
**PI:** Dr. Sudarsun Santhiappan | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

Computational analysis of classical poetic forms is a challenging NLP problem because it requires satisfying rigid formal constraints (meter, rhyme, cadence) while preserving semantic content. Venba's rules — Mathirai (syllabic weight), Thalai (syllable connections), Eerukural, Posai — constitute a context-free grammar that can be computationally validated but is much harder to generate from. This connects to the broader field of constrained text generation, where methods like constrained beam search (Anderson et al., 2017) and RL-based generation with constraint rewards (Khalifa et al., 2021) have been applied to tasks like keyword-constrained generation and format-controlled output.

The neuro-symbolic nature of this project — combining rule-based prosody checking with neural sequence-to-sequence models — represents a hybrid AI approach. My DL coursework covered seq2seq architectures (encoder-decoder with attention, and their transformer variants), and the ML course introduced the concept of incorporating domain constraints into optimisation objectives. The idea of using RL to fine-tune a generator towards satisfying grammatical constraints is an application of reward-shaped policy gradient methods.

The simplification task (Venba → prose) is a style transfer problem; the synthesis task (prose → Venba) is a constrained generation problem. Both require creating a parallel corpus, which is a data curation challenge that tests practical NLP skills. The dataset creation aspect (curating Venba-prose pairs) requires careful annotation methodology, connecting to my understanding of training data quality from the ML course.

---

## PROJECT 034 — Integrated ML and Dynamical System-based Optimisation for Safe Autonomy
**PI:** Prof. Arunkumar D Mahindrakar / Prof. Ramkrishna Pasumarthy | **Interns:** 1 | **Duration:** 6 months

**Statement of Purpose:**

Safe autonomous navigation requires guarantees that pure learning-based methods cannot currently provide. Control barrier functions (CBFs) offer formal safety certificates by defining invariant sets that the system must not leave, but they assume known dynamics. When dynamics are learned from data (using Gaussian processes or neural networks), the safety guarantee becomes probabilistic — and quantifying this probability requires careful analysis. The bridge between learning-based perception and provably safe control is one of the central open problems in robotics.

My ML coursework provides the learning-based component: understanding how function approximators (neural networks, GPs) learn from data, generalisation properties, and uncertainty quantification. Linear Algebra and Probability are directly relevant to the control and estimation theory that underpins the dynamical systems perspective — state-space representations, Lyapunov stability, and stochastic control all operate in this mathematical space. The strong mathematics requirement aligns with my coursework emphasis.

The implementation on TurtleBots grounds the theory in physical reality. Literature survey, modelling, analysis, simulation, and hardware implementation — this progression from theory to practice over 6 months is comprehensive. My interest in this project comes from the intersection it occupies: ML provides the adaptability, control theory provides the guarantees, and the challenge is making them work together without sacrificing either. The PI's existing work on optimisation with safety constraints using a dynamical systems perspective provides a clear research direction to build upon.

---

## PROJECT 035 — CareerSetu: AI-Based Guidance Co-Pilot
**PI:** Dr. S. Neethi / Dr. Sudarsun Santhiappan | **Interns:** 2–3 | **Duration:** 2–3 months

**Statement of Purpose:**

AI-based career guidance systems for K-12 education exist in various forms globally (Naviance, Kuder), but most are designed for Western education systems and do not account for India's specific landscape: multiple board systems (CBSE, state boards), ITI/polytechnic pathways alongside academic streams, state-specific scholarship structures, and the NSDC skill framework. Building a system that maps student profiles to these diverse pathways requires structured data curation before any AI recommendation logic can be applied.

My DBMS coursework is directly relevant to the data structuring tasks: compiling scholarship and scheme data from central and state sources into queryable databases with proper relational design. The ML coursework provides the foundation for the recommendation component — content-based filtering, rule-based systems, and potentially collaborative filtering approaches adapted for career path recommendation. Python is the implementation language for both data pipelines and prototype AI components.

The project explicitly welcomes interdisciplinary contributions and is described as research-and-design oriented rather than purely coding. The tasks include persona design, user journey mapping, and prototype development (conversational flows, recommendation logic, dashboards) — this combination of system design thinking with AI implementation is practically grounded. The emotion-aware component, while simple at the prototype stage, raises interesting questions about how recommendation systems should adapt when users express uncertainty — a problem that connects to contextual bandits and exploration-exploitation tradeoffs.

---

## PROJECT 036 — Legal NLP in India — An Exploratory Study
**PI:** Prof. B. Ravindran / Dr. Gokul S Krishnan | **Interns:** 1 | **Duration:** 3 months

**Statement of Purpose:**

Legal NLP has progressed rapidly in English-language jurisdictions through datasets like CaseHOLD, models like LegalBERT, and tasks like judgment prediction and statute retrieval. The Indian legal context is markedly different: multilingual proceedings, case backlog in the crores, inconsistent digitisation across courts, and a mix of common law precedent with constitutional directives. The few existing Indian legal NLP efforts (Indian Legal Documents Corpus, SCI-Summarizer) have scratched the surface but lack the breadth that a systematic exploratory study demands.

My DL and ML coursework covered the transformer architectures and fine-tuning methodologies that are central to building LM pipelines for legal tasks. The project requires building pipelines from scratch where applicable — this is where my hands-on coding experience matters. The image classification project demanded similar from-scratch implementation: data loading, model construction, training loops, evaluation. Transferring this implementation capability to NLP pipelines (tokenisation, embedding, classification/generation) is a shift in modality, not methodology.

The data extraction and curation aspect is often underestimated — working with messy, incomplete legal documents requires robust preprocessing pipelines and the willingness to spend time on data quality before model quality. My DBMS coursework provides the systematic thinking about data organisation that this requires. The potential for publication adds research structure to the work, requiring clear experimental design and reproducible methodology.

---

## PROJECT 037 — Evaluation Framework: Dynamic Evaluation & Evaluation Metrics
**PI:** Prof. B. Ravindran / Dr. Gokul S Krishnan | **Interns:** 1 | **Duration:** 3 months

**Statement of Purpose:**

The question of whether language models genuinely reason or merely retrieve memorised patterns from training data has become central to AI evaluation. Benchmark contamination — where test set examples leak into pre-training corpora — has been documented for GSM8K, MMLU, and other standard benchmarks. Dynamic evaluation, where test instances are generated or perturbed at evaluation time, addresses this by ensuring the model encounters genuinely novel inputs. Approaches range from template-based question generation to adversarial perturbation of existing benchmarks.

My ML coursework emphasised evaluation methodology: train-test splits, cross-validation, overfitting detection, and the distinction between memorisation and generalisation. This is exactly the conceptual framework needed here, scaled up to the LLM setting where "overfitting" manifests as benchmark memorisation rather than training loss divergence. The model evaluation experience from my image classification project — tracking training versus validation metrics, diagnosing whether the model learned features or shortcuts — provides practical grounding.

The project requires both LM pipeline building (constructing the evaluation infrastructure) and analytical thinking (designing metrics that distinguish reasoning from retrieval). Synthetic data generation — creating novel evaluation instances that test the same capabilities as existing benchmarks but with guaranteed novelty — is a creative and technical challenge. I am interested in exploring perturbation-based approaches: how much can you modify a benchmark question before the model's performance degrades, and what does the degradation curve reveal about the model's reasoning strategy?

---

## PROJECT 038 — An Exploratory Study on AI and Mental Health
**PI:** Prof. B. Ravindran / Dr. Gokul S Krishnan / Dr. Geetha Raju | **Interns:** 2 | **Duration:** 3 months (extendable to 6)

**Statement of Purpose:**

Reports of AI chatbot interactions contributing to mental health deterioration — including dependency formation, emotional manipulation through anthropomorphised responses, and inadequate crisis intervention — have raised urgent questions about the safety of conversational AI systems. The Tessa chatbot incident (NEDA, 2023) and subsequent analyses have shown that LLM-based systems can actively generate harmful dietary advice when prompted in certain ways, bypassing their safety training. The systematic study of how these systems behave in mental health contexts requires both technical evaluation (probing model responses under various scenarios) and policy analysis (what guardrails exist, what is missing).

My DL coursework provides the technical understanding of how these systems work — transformer architectures, RLHF alignment, safety tuning — which is necessary for understanding their failure modes. The ML course covered evaluation methodology that applies directly to designing systematic tests: defining test scenarios, measuring response quality across dimensions (safety, helpfulness, accuracy), and identifying systematic failure patterns. Python proficiency supports the LM pipeline building required for automated evaluation.

The dual nature of this project — technical evaluation combined with policy analysis — is distinctive. Understanding current AI and health policies across the globe (Task in the project description) requires the same systematic survey skills used in ML literature reviews. The goal of producing Dos and Don'ts for AI system usage is practically grounded and directly relevant. The extendable duration (3 to 6 months) suggests potential for deeper engagement if the initial work is productive.

---

## PROJECT 039 — Knowledge Graph Constrained Information Retrieval in Indic Languages
**PI:** Prof. G. Phanikumar | **Interns:** 2 | **Duration:** 2–4 months

**Statement of Purpose:**

Graph-RAG — using knowledge graph structure to constrain and guide LLM-based retrieval — addresses the hallucination problem by grounding generated responses in verified relational knowledge. The approach combines ontological reasoning (R-box axioms defining role properties like transitivity, symmetry, and composition) with neural retrieval, creating a hybrid system where the knowledge graph provides logical constraints and the LLM provides natural language fluency. Applying this to Indic languages adds the requirement of multilingual competence in a fully on-premise, disconnected setup.

My Graph Theory coursework provides direct preparation for understanding ontological structures: knowledge graphs are directed labelled graphs with type hierarchies, and R-box axioms define inference rules over these graphs. The OWL (Web Ontology Language) formalism that owlready2 implements is a description logic that operates on graph structures. My ML coursework covers the retrieval and embedding components of RAG systems, and Python is the implementation language.

The project emphasises computational efficiency — measuring power consumption with and without R-box axioms for specific inferences — which is an empirical evaluation of the computational cost of logical reasoning versus neural inference. This connects to questions about when structured knowledge (expensive to curate, cheap to query) outperforms unstructured neural retrieval (cheap to build, expensive to run). The on-premise, internet-disconnected requirement adds a practical constraint that favours smaller, locally deployed LLMs — a design consideration that shapes the entire architecture.

---

## PROJECT 040 — AI-for-VLSI Design
**PI:** Dr. Patanjali SLPSK | **Interns:** 2 | **Duration:** 3–6 months

**Statement of Purpose:**

The application of ML to VLSI design automation has been catalysed by Google's work on chip placement using reinforcement learning (Mirhoseini et al., 2021), which showed that RL agents can produce layouts competitive with human experts. Graph neural networks are a natural fit for representing netlists — where cells are nodes and wires are edges — and learning to predict placement quality or routing congestion from the graph structure. The EDA (Electronic Design Automation) pipeline has multiple optimisation stages (synthesis, placement, routing, timing closure) where ML can accelerate or improve traditional algorithms.

Graph Theory and ML/DL from my coursework map directly onto this project. GNN-based VLSI Place & Route operates on hypergraphs (nets connecting multiple pins), and understanding graph representations, adjacency structures, and message passing requires the foundation my Graph Theory course provides. The RL-based energy efficiency optimisation task requires understanding reward formulation, policy optimisation, and exploration strategies — concepts from my ML course extended to the sequential decision-making setting.

The PI lists the MIT Missing Semester course as a prerequisite for candidates lacking system skills — I have the required proficiency in Python, shell scripting, and version control. The note about patenting novel prototypes indicates the work aims for practical impact beyond publication. I am interested in how the GNN's inductive bias (local message passing) interacts with the global optimisation objective (minimise total wirelength or power), and whether hierarchical GNN architectures can capture multi-scale circuit structure.

---

## PROJECT 041 — AI for Cybersecurity
**PI:** Dr. Patanjali SLPSK | **Interns:** 2 | **Duration:** 3 months

**Statement of Purpose:**

RL-guided fuzzing has emerged as a promising direction for automated vulnerability discovery. Traditional fuzzers like AFL++ use coverage-guided mutation strategies with no explicit learning — the mutation selection is essentially random with coverage-based filtering. RL-based approaches (Böttinger et al., 2018; She et al., FuzzGuard) frame the mutation selection as a sequential decision problem where the agent learns which input regions and mutation operators are most likely to trigger new coverage or crashes. For firmware and hardware testing, the challenge is more constrained: execution environments require emulation (QEMU/Unicorn), the state space includes hardware register configurations, and bugs manifest as protocol violations rather than memory corruption.

My OS coursework provides the systems-level understanding needed for firmware analysis — process memory layout, interrupt handling, I/O mechanisms, and the kernel-userspace boundary. Computer Networks covers the protocol stack relevant to hardware-software interfaces. Python and C/C++ are the required languages, both within my coursework. The RL algorithm development connects to my ML coursework — defining state representations, action spaces (mutation operators), and reward signals (coverage gain, crash detection) for the fuzzing agent.

The dual focus — firmware fuzzing and hardware-software testing — provides exposure to both software and hardware security, which are distinct problem domains unified by the RL methodology. The prospect of contributing to research papers and patents, combined with the career relevance for semiconductor and cybersecurity roles, makes this project practically valuable beyond the research experience itself.

