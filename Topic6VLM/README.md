# Vision-Language Agents with LangGraph & LLaVA

This project contains two Vision-Language AI systems:

- Exercise 1 – Multi-turn Image Chat Agent (Tkinter Desktop App)
- Exercise 2 – Video Surveillance Person Detection 

Both use LLaVA via Ollama.

---

# Table of Contents

- [Project Structure](#project-structure)
- [Exercise 1 – Vision-Language Chat Agent](#exercise-1--vision-language-chat-agent)
  - [What It Does](#what-it-does)
  - [How To Run](#how-to-run)
- [Exercise 2 – Video Surveillance](#exercise-2--video-surveillance)
  - [What It Does](#what-it-does-1)
  - [How To Run](#how-to-run-1)
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)

---

# Project Structure

```
Topic6VLM/
│
├── Exercise 1/
│   ├── code/
│   │   └── vision_chat_agent.py
│   ├── img/
│   │   ├── landscape_resized_216.jpg
│   │   └── landscape.jpg
│   └── output/
│       └── vlm_checkpoints.db
│
├── Exercise 2/
│   ├── code/
│   │   └── video_surveillance.py
│   ├── output/
│   │   └── person_times.txt
│   └── video/
│       └── person.mp4
│
└── README.md
```

---

# Exercise 1 – Vision-Language Chat Agent

## What It Does

This program:

- Opens a desktop application using Tkinter.
- Allows you to load an image.
- Lets you ask multiple questions about the image.
- Maintains conversation memory across turns.
- Stores conversation state in a local database.
- Continues the discussion using previous context.

The conversation persists during the session.

---

## How To Run

### Install dependencies

```bash
pip install ollama langgraph langchain-core pillow
```

---

### Pull the required model

```bash
ollama pull llava:7b-v1.6-mistral-q4_0
```

---

### Run the Tkinter application

```bash
python "Exercise 1\code\vision_chat_agent.py" --ui tk
```

A desktop window will open.

Steps:
1. Click **Load Image**
2. Select an image
3. Ask questions in the text box
4. Press Enter to send

---

# Exercise 2 – Video Surveillance

## What It Does

This program:

- Takes a video file as input.
- Extracts frames every 2 seconds.
- Checks each frame for the presence of a person.
- Detects when a person:
  - Enters the scene
  - Exits the scene
- Writes detected entry and exit times to a text file.

It analyzes the video by processing sampled frames sequentially.

---

## How To Run

### Install dependencies

```bash
pip install opencv-python ollama pillow
```

### Run the program

```bash
python "Exercise 2\code\video_surveillance.py" --video "Exercise 2\video\person.mp4"
```

### Optional: Faster Mode

```bash
python "Exercise 2\code\video_surveillance.py" --video "Exercise 2\video\person.mp4" --use_local_detector
```

---

## Output

Results are saved to:

```
Exercise 2\output\person_times.txt
```

Example output:

```
ENTER: 12.00s    EXIT: 46.00s
```

---

# Dependencies

Install everything at once:

```bash
pip install ollama langgraph langchain-core pillow opencv-python
```

---

# Model Setup

Model used:

```
llava:7b-v1.6-mistral-q4_0
```

If not installed:

```bash
ollama pull llava:7b-v1.6-mistral-q4_0
```

---

# Summary

| Exercise | Function |
|-----------|----------|
| Exercise 1 | Desktop multi-turn image chat agent |
| Exercise 2 | Video person entry/exit detection |

---
