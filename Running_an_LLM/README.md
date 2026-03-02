# Vision-Language Agents with LangGraph & LLaVA

The goal of this project was to evaluate multiple small and medium-sized large language models on selected MMLU subjects, compare inference performance under different hardware and quantization configurations, analyze model mistake patterns, and build a custom chat agent from scratch.

---

# Table of Contents

1. [Project Directory Structure](#project-directory-structure)  
2. [Verification of Setup](#verification-of-setup)  
3. [Timing Experiments](#timing-experiments)  
4. [10-Subject Evaluation with Models](#10-subject-evaluation-with-models)    
5. [Chat Agent Implementation](#chat-agent-implementation)  
6. [Conclusion](#conclusion)  

---

# Project Directory Structure

```
.
├── Task 3 - Testing Code Output/
│   ├── output.txt
│  
│
├── Task 4 - Timing Code Output/
│   ├── output_cpu_no_quant_local.txt
│   ├── output_cpu_4_bit_quant_local.txt
│   ├── output_gpu_no_quant_colab.txt
│   ├── output_gpu_4_bit_quant_colab.txt
│   ├── output_gpu_8_bit_quant_colab.txt
│
├── Task 5,6 and 7 - Modifying Code/│  
│   ├── 3_models_accuracy_by_subject.png
│   ├── 6_models_accuracy_by_subject.png
│   ├── accuracy_3_models.png
│   ├── accuracy_6_models.png
│   ├── time_cpu.png
│   ├── times_gpu.png
│   
│
├── Task 8 - Chat Agent Output/│
│   ├── flag_true_output.txt
│   ├── flag_false_output.txt
│   ├── simple_chat_agent_output.txt
│   ├── output_context_window.txt
│
├── analysis.pdf
├── llama_mmlu_eval.py
├── simple_chat_agent.ipynb
├── chat_agent_context_window.ipynb
├── chat_agent_flag_off.ipynb
├── three_models_ten_subject_eval.py
├── six_models_ten_subject_eval.py
├── plot.py
├── README.md
```

---

# Verification of Setup

The file `llama_mmlu_eval.py` was executed on two MMLU subjects to confirm that the setup was working correctly.

The script successfully:
- Loaded the model
- Loaded the dataset
- Generated predictions
- Computed accuracy
- Logged timing information

This confirmed that the environment was correctly configured.

---

# Timing Experiments

The `time` shell command was used to measure execution time for different configurations.

## CPU and No Quantization

```
real    1m50.849s
user    21m37.960s
sys     0m4.498s
```

The CPU execution without quantization was significantly slow because all computations were performed in full precision on the processor.


## CPU with 4-bit Quantization

```
real    0m21.355s
user    0m21.925s
sys     0m1.771s
```

Using 4-bit quantization on CPU drastically reduced runtime. The reduction in memory and computation precision led to much faster inference.


## GPU and No Quantization

```
real    0m25.811s
user    0m23.948s
sys     0m3.778s
```

Running the model on GPU without quantization provided a substantial speed improvement compared to CPU full precision.


## GPU with 4-bit Quantization

```
real    0m50.708s
user    0m44.956s
sys     0m12.328s
```

Interestingly, GPU inference with 4-bit quantization was slower than full precision GPU inference. This occurred because small models do not benefit significantly from quantization overhead on GPU.

## GPU with 8-bit Quantization

```
real    0m52.555s
user    0m51.014s
sys     0m14.704s
```

The 8-bit configuration was also slower than full precision GPU inference for this model size.

## Timing Conclusion

The results show that:

- CPU benefits significantly from 4-bit quantization.
- GPU performs best without quantization for small models.
- Quantization benefits increase for larger models.

---

# 10-Subject Evaluation with Models

## Local

Three models were evaluated on ten selected MMLU subjects:

- Llama-3.2-1B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-0.5B-Instruct

Accuracy and timing were logged per model, per subject, and per question.

The overall result showed that Llama-3.2-1B achieved higher accuracy than the Qwen models across most subjects.

Business-related subjects had higher accuracy, while formal logic and abstract algebra had lower accuracy across all models.

## Google Colab

In Google Colab, three additional models were evaluated:

- Qwen2.5-7B-Instruct
- OLMo-2-0425-1B
- Mistral-7B-v0.1

The 7B models significantly outperformed the 1B models in most STEM subjects.

This demonstrates that larger model capacity improves reasoning and domain knowledge performance.

## Performance Graphs

The following graphs were generated:

- `3_models_accuracy_by_subject.png`
- `6_models_accuracy_by_subject.png`
- `accuracy_3_models.png`
- `accuracy_6_models.png`
- `time_cpu.png`
- `times_gpu.png`

The graphs show a clear trend that performance increases with model size.


## Analysis of Model Mistakes

The mistakes were not random.

The following patterns were observed:

1. All models struggled with formal logic and symbolic reasoning.
2. Physics-based astronomy questions caused confusion.
3. Business and marketing subjects had consistently higher accuracy.
4. Smaller models often selected the wrong letter even when the explanation text was correct.
5. Larger models made fewer formatting-related mistakes.

Additionally, some difficult STEM questions were missed by all models, suggesting common reasoning limitations.

---

# Chat Agent Implementation

A custom chat agent was implemented without using a pre-built chat framework.

The agent:

- Loads the model
- Maintains conversation history
- Tracks token usage
- Prints latency
- Allows exit command

The implementation demonstrates how conversational context is constructed and fed back into the model.


## Context Management Strategy

The initial implementation allowed unlimited context growth.
To prevent context overflow, a context window management strategy was implemented.
The strategy keeps only the most recent conversation turns within the token limit. This prevents memory overflow and ensures long conversations do not crash.


## Conversation History Flag Comparison

A flag was added to toggle conversation history.
When history was enabled, the model correctly remembered the user’s favorite animal and their name.
When history was disabled, the model forgot previous information and hallucinated incorrect answers.
This experiment demonstrates that small models rely heavily on conversation history for multi-turn coherence.

---

# Conclusion

This project demonstrates that:

- GPU inference significantly outperforms CPU full precision.
- CPU 4-bit quantization provides major speed improvements.
- GPU quantization does not help small models.
- Larger models consistently outperform smaller models.
- Errors are systematic rather than random.
- Context retention is critical for multi-turn chat performance.
- Managing the context window is necessary for stable long conversations.

Overall, the experiments show that model size, hardware configuration, and context management all significantly influence LLM performance.

---