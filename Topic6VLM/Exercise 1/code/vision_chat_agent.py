import argparse
import os
import threading
from typing import Annotated, Optional, Sequence, TypedDict
import ollama
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, scrolledtext


MODEL = "llava:7b-v1.6-mistral-q4_0"
MAX_SIDE = 216
CONTEXT_MESSAGES = 2
CHECKPOINT_DB = "vlm_checkpoints.db"
THREAD_ID = "vlm_thread"

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "Answer the user's questions about the provided image. "
    "Use the conversation history for context. Be concise but specific."
)


class VLMState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    user_input: str
    image_path: Optional[str]
    should_exit: bool
    verbose: bool
    reprompt: bool


_resize_cache: dict[str, str] = {}


def maybe_resize(image_path: str, max_side: int = MAX_SIDE) -> str:
    cache_key = f"{image_path}@{max_side}"
    if cache_key in _resize_cache:
        return _resize_cache[cache_key]

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) <= max_side:
        _resize_cache[cache_key] = image_path
        return image_path

    scale = max_side / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = image_path.rsplit(".", 1)[0] + f"_resized_{max_side}.jpg"
    img.save(out, format="JPEG", quality=90)
    _resize_cache[cache_key] = out
    return out


def build_ollama_messages(messages: Sequence[AnyMessage], image_path: str):
    recent = list(messages)[-CONTEXT_MESSAGES:]
    last_human_idx = max(
        (i for i, m in enumerate(recent) if isinstance(m, HumanMessage)),
        default=None,
    )

    ollama_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, m in enumerate(recent):
        if isinstance(m, HumanMessage):
            entry = {"role": "user", "content": m.content}
            if image_path and i == last_human_idx:
                entry["images"] = [image_path]
            ollama_msgs.append(entry)
        elif isinstance(m, AIMessage):
            ollama_msgs.append({"role": "assistant", "content": m.content})

    return ollama_msgs


def ingest_user_turn(state: VLMState) -> dict:
    text = (state.get("user_input") or "").strip()

    if text.lower() in ("quit", "exit", "q"):
        return {"should_exit": True}

    if text.lower() == "verbose":
        return {"verbose": True, "reprompt": True}

    if text.lower() == "quiet":
        return {"verbose": False, "reprompt": True}

    if text == "":
        return {"reprompt": True}

    return {"reprompt": False, "messages": [HumanMessage(content=text)]}


def call_llava(state: VLMState) -> dict:
    img_path = state.get("image_path") or ""
    messages = state.get("messages", [])

    if not img_path or not os.path.isfile(img_path):
        return {"messages": [AIMessage(content="Please load an image first.")]}

    img_to_use = maybe_resize(img_path, MAX_SIDE)
    ollama_msgs = build_ollama_messages(messages, img_to_use)

    resp = ollama.chat(model=MODEL, messages=ollama_msgs)
    answer = resp["message"]["content"].strip()

    return {"messages": [AIMessage(content=answer)]}


def build_graph(checkpointer):
    g = StateGraph(VLMState)
    g.add_node("ingest_user_turn", ingest_user_turn)
    g.add_node("call_llava", call_llava)

    g.add_edge(START, "ingest_user_turn")
    g.add_edge("ingest_user_turn", "call_llava")
    g.add_edge("call_llava", END)

    return g.compile(checkpointer=checkpointer)


# ===================== TKINTER UI =====================

def launch_tkinter(checkpointer):
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    root = tk.Tk()
    root.title("LLaVA Vision Chat")
    root.geometry("800x700")

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)

    img_label = tk.Label(top, text="No image loaded", width=40, anchor="w")
    img_label.pack(side="left", padx=(0, 8))

    img_preview_label = tk.Label(top)
    img_preview_label.pack(side="left")

    history = scrolledtext.ScrolledText(root, wrap="word", height=25, state="disabled")
    history.pack(fill="both", expand=True, padx=8, pady=6)

    def append_history(role: str, text: str):
        history.configure(state="normal")
        prefix = "You: " if role == "user" else "Assistant: "
        history.insert("end", f"{prefix}{text}\n\n")
        history.see("end")
        history.configure(state="disabled")

    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=8, pady=6)

    entry_var = tk.StringVar()
    entry = tk.Entry(bottom, textvariable=entry_var)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    entry.focus_set()

    send_btn = tk.Button(bottom, text="Send")
    send_btn.pack(side="right")

    def on_load_image():
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if not path:
            return

        init: VLMState = {
            "messages": [],
            "user_input": "",
            "image_path": path,
            "should_exit": False,
            "verbose": False,
            "reprompt": False,
        }

        graph.invoke(init, config=config)
        img_label.config(text=os.path.basename(path))

        pil_img = Image.open(path).convert("RGB")
        pil_img.thumbnail((120, 120))
        tk_img = ImageTk.PhotoImage(pil_img)
        img_preview_label.image = tk_img
        img_preview_label.config(image=tk_img)

    def on_send(event=None):
        text = entry_var.get().strip()
        if not text:
            return

        append_history("user", text)
        entry_var.set("")

        current = graph.get_state(config).values or {}

        update: VLMState = {
            **current,
            "user_input": text,
            "should_exit": False,
            "reprompt": False,
        }

        out = graph.invoke(update, config=config)

        reply = ""
        for m in reversed(out.get("messages", [])):
            if isinstance(m, AIMessage):
                reply = m.content
                break

        append_history("assistant", reply)

    send_btn.config(command=on_send)
    load_btn = tk.Button(top, text="Load Image", command=on_load_image)
    load_btn.pack(side="right")

    root.bind("<Return>", on_send)
    root.mainloop()


# ===================== CLI =====================

def launch_cli(checkpointer):
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("Vision-Language Chat Agent (CLI)")
    print("Type 'quit' to exit")

    state: VLMState = {
        "messages": [],
        "user_input": "",
        "image_path": "",
        "should_exit": False,
        "verbose": False,
        "reprompt": False,
    }

    image_path = input("Enter image path: ").strip()
    state["image_path"] = image_path

    graph.invoke(state, config=config)

    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit"):
            break

        current = graph.get_state(config).values or {}

        update: VLMState = {
            **current,
            "user_input": text,
            "should_exit": False,
            "reprompt": False,
        }

        out = graph.invoke(update, config=config)

        for m in reversed(out.get("messages", [])):
            if isinstance(m, AIMessage):
                print("Assistant:", m.content)
                break



def main():
    parser = argparse.ArgumentParser(description="LLaVA Vision Chat")
    parser.add_argument("--ui", choices=["tk", "cli"], default="tk")
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        if args.ui == "cli":
            launch_cli(checkpointer)
        else:
            launch_tkinter(checkpointer)


if __name__ == "__main__":
    main()