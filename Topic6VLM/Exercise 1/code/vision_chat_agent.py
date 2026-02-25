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

try:
    import gradio as gr
    HAS_GRADIO = True
except Exception:
    HAS_GRADIO = False

try:
    import tkinter as tk
    from tkinter import filedialog, scrolledtext
    HAS_TK = True
except Exception:
    HAS_TK = False

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


def build_ollama_messages(messages: Sequence[AnyMessage], image_path: str, context_n: int = CONTEXT_MESSAGES) -> list:
    recent = list(messages)[-context_n:]
    last_human_idx = max((i for i, m in enumerate(recent) if isinstance(m, HumanMessage)), default=None)
    ollama_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, m in enumerate(recent):
        if isinstance(m, HumanMessage):
            entry: dict = {"role": "user", "content": m.content}
            if image_path and i == last_human_idx:
                entry["images"] = [image_path]
            ollama_msgs.append(entry)
        elif isinstance(m, AIMessage):
            ollama_msgs.append({"role": "assistant", "content": m.content})
    return ollama_msgs

def ingest_user_turn(state: VLMState) -> dict:
    verbose = state.get("verbose", False)
    text = (state.get("user_input") or "").strip()
    if verbose:
        print(f"\n[TRACE] ingest_user_turn  user_input='{text}'")
    if text.lower() in ("quit", "exit", "q"):
        return {"should_exit": True, "reprompt": False}
    if text.lower() == "verbose":
        print("Verbose mode ON")
        return {"verbose": True, "reprompt": True}
    if text.lower() == "quiet":
        return {"verbose": False, "reprompt": True}
    if text == "":
        return {"reprompt": True}
    return {"reprompt": False, "messages": [HumanMessage(content=text)]}


def call_llava(state: VLMState) -> dict:
    verbose = state.get("verbose", False)
    img_path = state.get("image_path") or ""
    messages = state.get("messages", [])
    if verbose:
        print(f"\n[TRACE] call_llava  messages={len(messages)}  image={img_path}")
    if not img_path or not os.path.isfile(img_path):
        return {"messages": [AIMessage(content="Please upload an image first.")]}
    img_to_use = maybe_resize(img_path, MAX_SIDE) if MAX_SIDE else img_path
    ollama_msgs = build_ollama_messages(messages, img_to_use)
    if verbose:
        print(f"[TRACE] Sending {len(ollama_msgs)} messages to Ollama ({MODEL})")
    resp = ollama.chat(model=MODEL, messages=ollama_msgs)
    answer = resp["message"]["content"].strip()
    if verbose:
        print(f"[TRACE] LLaVA reply: {answer[:120]}")
    return {"messages": [AIMessage(content=answer)]}


def print_response(state: VLMState) -> dict:
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, AIMessage):
            print("\n" + "─" * 60)
            print("LLaVA:", m.content)
            print("─" * 60)
            break
    return {}


def get_cli_input(state: VLMState) -> dict:
    print("\n" + "═" * 60)
    if not state.get("image_path"):
        print("No image loaded. Enter a file path to an image:")
    else:
        print("Ask a question (or 'quit' / 'verbose' / 'quiet'):")
    print("═" * 60)
    print("> ", end="", flush=True)
    text = input().strip()
    if not state.get("image_path"):
        if text.lower() in ("quit", "exit", "q"):
            return {"should_exit": True, "reprompt": False}
        if not os.path.isfile(text):
            print(f"  File not found: {text}")
            return {"reprompt": True}
        print(f"Image loaded: {text}")
        return {"image_path": text, "reprompt": True, "user_input": ""}
    return {"user_input": text, "reprompt": False}

def route_after_ingest_cli(state: VLMState) -> str:
    if state.get("should_exit"):
        return END
    if state.get("reprompt"):
        return "get_cli_input"
    return "call_llava"


def route_after_ingest_gradio(state: VLMState) -> str:
    if state.get("should_exit") or state.get("reprompt"):
        return END
    return "call_llava"


def route_after_print(state: VLMState) -> str:
    if state.get("should_exit"):
        return END
    return "get_cli_input"


def build_cli_graph(checkpointer):
    g = StateGraph(VLMState)
    g.add_node("get_cli_input", get_cli_input)
    g.add_node("ingest_user_turn", ingest_user_turn)
    g.add_node("call_llava", call_llava)
    g.add_node("print_response", print_response)
    g.add_edge(START, "get_cli_input")
    g.add_edge("get_cli_input", "ingest_user_turn")
    g.add_conditional_edges("ingest_user_turn", route_after_ingest_cli,
                            {"get_cli_input": "get_cli_input", "call_llava": "call_llava", END: END})
    g.add_edge("call_llava", "print_response")
    g.add_conditional_edges("print_response", route_after_print,
                            {"get_cli_input": "get_cli_input", END: END})
    return g.compile(checkpointer=checkpointer)


def build_gradio_graph(checkpointer):
    g = StateGraph(VLMState)
    g.add_node("ingest_user_turn", ingest_user_turn)
    g.add_node("call_llava", call_llava)
    g.add_edge(START, "ingest_user_turn")
    g.add_conditional_edges("ingest_user_turn", route_after_ingest_gradio,
                            {"call_llava": "call_llava", END: END})
    g.add_edge("call_llava", END)
    return g.compile(checkpointer=checkpointer)


def launch_gradio(checkpointer, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    if not HAS_GRADIO:
        raise RuntimeError("Gradio is not installed.")
    graph = build_gradio_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    def _last_ai(messages) -> str:
        for m in reversed(list(messages)):
            if isinstance(m, AIMessage):
                return m.content
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
            if isinstance(m, str):
                return m
        return "(No response)"

    def set_image(img_filepath):
        if not img_filepath:
            return "No image selected.", []
        init: VLMState = {"messages": [], "user_input": "", "image_path": img_filepath,
                          "should_exit": False, "verbose": False, "reprompt": False}
        graph.invoke(init, config=config)
        return f"Image loaded: {os.path.basename(img_filepath)}", []

    def _normalize_to_message_dicts(chat_history):
        out: list[dict] = []
        if not chat_history:
            return out
        for item in chat_history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                out.append({"role": str(item["role"]), "content": "" if item["content"] is None else str(item["content"])})
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_part = "" if item[0] is None else str(item[0])
                assistant_part = "" if item[1] is None else str(item[1])
                if user_part != "":
                    out.append({"role": "user", "content": user_part})
                if assistant_part != "":
                    out.append({"role": "assistant", "content": assistant_part})
                continue
            try:
                from langchain_core.messages import HumanMessage as _HM, AIMessage as _AM
            except Exception:
                _HM = _AM = None
            if _HM and isinstance(item, _HM):
                out.append({"role": "user", "content": str(item.content)})
                continue
            if _AM and isinstance(item, _AM):
                out.append({"role": "assistant", "content": str(item.content)})
                continue
            if isinstance(item, str):
                out.append({"role": "assistant", "content": item})
                continue
            out.append({"role": "assistant", "content": str(item)})
        return out

    def chat(user_text: str, chat_history: list):
        safe_msgs = _normalize_to_message_dicts(chat_history)
        text = (user_text or "").strip()
        if not text:
            return safe_msgs, gr.update(value="")
        if text.lower() in ("verbose", "quiet"):
            current = graph.get_state(config).values or {}
            update: VLMState = {**current, "user_input": text, "should_exit": False, "reprompt": False}
            graph.invoke(update, config=config)
            label = ("🔍 Verbose mode ON — tracing will appear in server logs." if text.lower() == "verbose"
                     else "🔇 Quiet mode ON — tracing suppressed.")
            safe_msgs.append({"role": "user", "content": text})
            safe_msgs.append({"role": "assistant", "content": label})
            return safe_msgs, gr.update(value="")
        current = graph.get_state(config).values or {}
        img_path = current.get("image_path", "")
        verbose = current.get("verbose", False)
        update: VLMState = {**current, "user_input": text, "image_path": img_path, "verbose": verbose,
                            "should_exit": False, "reprompt": False}
        out = graph.invoke(update, config=config)
        reply = _last_ai(out.get("messages", []))
        safe_msgs.append({"role": "user", "content": text})
        safe_msgs.append({"role": "assistant", "content": reply})
        return safe_msgs, gr.update(value="")

    with gr.Blocks(title="LLaVA Vision Chat") as demo:
        gr.Markdown(
            "## Vision-Language Chat Agent\n"
            "Upload an image, click **Load Image**, then ask questions. "
            "Conversation history is preserved across turns.\n\n"
            "_Type_ `verbose` _or_ `quiet` _in the chat to toggle tracing._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="filepath", label="Upload Image")
                set_btn = gr.Button("Load Image", variant="primary")
                img_status = gr.Textbox(label="Status", interactive=False, lines=1)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    msg_box = gr.Textbox(placeholder="Ask something about the image…", show_label=False, scale=8)
                    send_btn = gr.Button("Send", scale=1, variant="primary")

        set_btn.click(set_image, inputs=img_input, outputs=[img_status, chatbot])
        send_btn.click(chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
        msg_box.submit(chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])

    demo.launch(server_name=host, server_port=port, share=share)

def launch_tkinter(checkpointer):
    if not HAS_TK:
        raise RuntimeError("tkinter not available in this Python environment.")
    graph = build_gradio_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    root = tk.Tk()
    root.title("LLaVA Vision Chat (Tkinter)")
    root.geometry("800x700")

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)

    img_label = tk.Label(top, text="No image loaded", width=40, anchor="w")
    img_label.pack(side="left", padx=(0, 8))

    img_preview_label = tk.Label(top)
    img_preview_label.pack(side="left")

    status_label = tk.Label(top, text="", width=30, anchor="w")
    status_label.pack(side="right", padx=(6, 0))

    history = scrolledtext.ScrolledText(root, wrap="word", height=25, state="disabled")
    history.pack(fill="both", expand=True, padx=8, pady=6)

    def append_history(role: str, text: str):
        history.configure(state="normal")
        if role == "user":
            history.insert("end", f"You: {text}\n")
        else:
            history.insert("end", f"Assistant: {text}\n\n")
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

    def _last_ai(messages):
        for m in reversed(list(messages)):
            if isinstance(m, AIMessage):
                return str(m.content)
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
            if isinstance(m, str):
                return m
        return "(No response)"

    current_image_path = {"path": None}

    def on_load_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                                                     ("All files", "*.*")])
        if not path:
            return
        init: VLMState = {"messages": [], "user_input": "", "image_path": path,
                          "should_exit": False, "verbose": False, "reprompt": False}
        graph.invoke(init, config=config)
        current_image_path["path"] = path
        img_label.config(text=os.path.basename(path))
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_img.thumbnail((120, 120))
            tk_img = ImageTk.PhotoImage(pil_img)
            img_preview_label.image = tk_img
            img_preview_label.config(image=tk_img)
        except Exception:
            img_preview_label.config(text="[preview failed]")

    def invoke_graph_in_thread(user_text: str):
        def worker():
            try:
                current = graph.get_state(config).values or {}
                img_path = current.get("image_path", "")
                verbose = current.get("verbose", False)
                update: VLMState = {**current, "user_input": user_text, "image_path": img_path, "verbose": verbose,
                                    "should_exit": False, "reprompt": False}
                out = graph.invoke(update, config=config)
                reply = _last_ai(out.get("messages", []))
            except Exception as e:
                reply = f"[error] {e}"
            root.after(0, lambda: append_history("assistant", reply))
            root.after(0, lambda: status_label.config(text=""))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def on_send(event=None):
        text = entry_var.get().strip()
        if not text:
            return
        append_history("user", text)
        entry_var.set("")
        status_label.config(text="Thinking...")
        if text.lower() in ("verbose", "quiet"):
            current = graph.get_state(config).values or {}
            update: VLMState = {**current, "user_input": text, "should_exit": False, "reprompt": False}
            graph.invoke(update, config=config)
            label = ("Verbose mode ON — check console." if text.lower() == "verbose" else "Quiet mode ON")
            append_history("assistant", label)
            status_label.config(text="")
            return
        invoke_graph_in_thread(text)

    send_btn.config(command=on_send)
    load_btn = tk.Button(top, text="Load Image", command=on_load_image)
    load_btn.pack(side="right")
    root.bind("<Return>", on_send)
    root.mainloop()

def launch_cli(checkpointer, preload_image: str = ""):
    graph = build_cli_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}
    print("=" * 60)
    print("  Vision-Language Chat Agent  (LLaVA via Ollama)")
    print("  Commands: verbose | quiet | quit")
    print("=" * 60)
    saved = graph.get_state(config)
    if saved.next:
        print("Resuming previous session...")
        graph.invoke(None, config=config)
        return
    init: VLMState = {"messages": [], "user_input": "", "image_path": preload_image if os.path.isfile(preload_image) else "",
                      "should_exit": False, "verbose": False, "reprompt": False}
    if preload_image and os.path.isfile(preload_image):
        print(f"Pre-loaded image: {preload_image}")
    graph.invoke(init, config=config)
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="LLaVA Vision-Language LangGraph Chat")
    parser.add_argument("--ui", choices=["gradio", "cli", "tk"],
                        default="gradio" if HAS_GRADIO else ("tk" if HAS_TK else "cli"),
                        help="Interface mode (gradio | tk | cli)")
    parser.add_argument("--image", default="", help="Pre-load an image file path (CLI mode only)")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio host (if using gradio)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port (if using gradio)")
    parser.add_argument("--share", action="store_true", help="Use Gradio share link (if using gradio)")
    args, _unknown = parser.parse_known_args()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        if args.ui == "gradio":
            if not HAS_GRADIO:
                print("Gradio is not installed — pip install gradio")
                print("Falling back to CLI mode.")
                launch_cli(checkpointer, preload_image=args.image)
            else:
                # pass host/port/share into launch_gradio by setting local names used in function
                global host, port, share
                host, port, share = args.host, args.port, args.share
                launch_gradio(checkpointer, host=args.host, port=args.port, share=args.share)
        elif args.ui in ("tk", "tkinter"):
            launch_tkinter(checkpointer)
        else:
            launch_cli(checkpointer, preload_image=args.image)


if __name__ == "__main__":
    main()