#!/usr/bin/env python3
"""Local Gradio frontend for SmartDesk. Save as scripts/smartdesk_web.py.

Uses the same LangGraph workflow as smartdesk_chat.py. Each browser session
gets its own graph thread. The UI translates form actions into graph resumes;
the graph still owns retrieval, ticket lookups, and approval before writes.
"""

import argparse
import os
from uuid import uuid4

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from smartdesk_chat import FDE_DIR, PROJECT_DIR, build_graph, normalize_email
from tickets import validate_draft


def new_session():
    return {
        "config": {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 100},
        "chat": [],
        "shown": set(),
        "pending": None,
    }


def chat_reply(text, pending):
    """Return a resume value for interrupts that accept a chat reply."""
    kind = pending["kind"]
    if kind == "offer":
        if text.lower() in {"yes", "y", "no", "n", "/cancel"}:
            return "yes" if text.lower() in {"yes", "y"} else "no"
        raise ValueError("Reply yes or no.")
    if kind == "email":
        return {"cancel": True} if text.lower() == "/cancel" else normalize_email(text)
    if kind == "select":
        if text.lower() == "/cancel":
            return {"cancel": True}
        if text.removeprefix("#") in {str(ticket["number"]) for ticket in pending["tickets"]}:
            return int(text.removeprefix("#"))
        raise ValueError("Enter a ticket number from the list, or /cancel.")
    raise ValueError("Use the buttons in the ticket preview.")


def advance(graph, session, action, text="", summary="", description="", category=""):
    """Advance one graph step and collect only newly emitted messages."""
    pending = session["pending"]
    if action == "send":
        text = (text or "").strip()
        if not text:
            return "Enter a message first."
        if pending:
            try:
                command = Command(resume=chat_reply(text, pending))
            except ValueError as error:
                return str(error)
        else:
            command = {"messages": [HumanMessage(content=text)]}
        session["chat"].append({"role": "user", "content": text})
    else:
        if not pending or pending["kind"] != "approval":
            return "There is no ticket waiting for approval."
        draft = {"summary": summary or "", "description": description or "", "category": category or ""}
        if action == "approve" and draft != pending["draft"]:
            return "Apply edits and review the updated preview before creating the ticket."
        if action in {"edit", "approve"}:
            try:
                validate_draft(pending["email"], **draft)
            except ValueError as error:
                return str(error)
        command = Command(
            resume={"action": "edit", "draft": draft}
            if action == "edit"
            else {"action": action}
        )

    result = graph.invoke(command, config=session["config"])
    for message in result.get("messages", []):
        if message.type == "ai" and message.id not in session["shown"]:
            session["chat"].append({"role": "assistant", "content": message.content})
            session["shown"].add(message.id)
    interrupts = result.get("__interrupt__", ())
    session["pending"] = interrupts[0].value if interrupts else None
    return ""


def build_app(graph):
    """Display the graph's pending question or approval form."""

    def clear_checkpoint(session):
        if session:
            thread_id = session["config"]["configurable"]["thread_id"]
            graph.checkpointer.delete_thread(thread_id)

    def render(session, status="", text="", unsaved=None):
        pending = session["pending"] or {}
        kind = pending.get("kind")
        approval = kind == "approval"
        draft = unsaved if approval and unsaved is not None else pending.get("draft", {})
        prompt = pending.get("prompt", "")
        if kind == "offer":
            prompt += " (yes/no)"
        if kind == "email":
            prompt += " (or /cancel)"
        if kind == "select":
            prompt = "Choose a ticket number, or /cancel:\n" + "\n".join(
                f"#{ticket['number']}: {ticket['title']} — {ticket['state']}"
                for ticket in pending["tickets"]
            )
        return (
            session,
            list(session["chat"]),
            gr.update(value=text, interactive=not approval),
            gr.update(interactive=not approval),
            gr.update(value=prompt, visible=bool(prompt)),
            gr.update(visible=approval),
            pending.get("email", ""),
            draft.get("summary", ""),
            draft.get("description", ""),
            draft.get("category"),
            status,
        )

    def handle(action):
        def callback(session, text, summary, description, category):
            session = session or new_session()
            if action == "reset":
                clear_checkpoint(session)
                return render(new_session())
            try:
                status = advance(graph, session, action, text, summary, description, category)
            except Exception:
                # The write might already have reached GitHub. Never replay it here.
                clear_checkpoint(session)
                status = "The turn failed. Check GitHub before retrying a ticket submission."
                return render(new_session(), status)
            return render(
                session, status, text if status and action == "send" else "",
                {"summary": summary, "description": description, "category": category}
                if status and action in {"edit", "approve"} else None,
            )
        return callback

    with gr.Blocks(title="SmartDesk", analytics_enabled=False) as app:
        session = gr.State(value=None, time_to_live=3600, delete_callback=clear_checkpoint)
        gr.Markdown(
            "# SmartDesk\n"
            "Ask a NovaTech IT or HR question, request a support ticket, or check ticket status."
        )
        chat = gr.Chatbot(label="Conversation", render_markdown=False, height=440)
        prompt = gr.Textbox(label="SmartDesk needs your reply", visible=False, interactive=False)
        with gr.Row():
            message = gr.Textbox(label="Message", placeholder="Ask SmartDesk…", scale=5)
            send = gr.Button("Send", variant="primary")
            reset = gr.Button("New chat")
        with gr.Group(visible=False) as preview:
            gr.Markdown("### Review ticket\nApply any edits before creating the ticket.")
            email = gr.Textbox(label="Employee email", interactive=False)
            summary = gr.Textbox(label="Summary")
            description = gr.Textbox(label="Description", lines=4)
            category = gr.Dropdown(["IT", "HR"], label="Category")
            with gr.Row():
                edit = gr.Button("Apply edits")
                create = gr.Button("Create GitHub issue", variant="primary")
                cancel = gr.Button("Cancel")
        status = gr.Textbox(label="Status", interactive=False)
        inputs = [session, message, summary, description, category]
        outputs = [session, chat, message, send, prompt, preview, email,
                   summary, description, category, status]

        # Serialize graph calls, including form actions and resets.
        for event, action in ((message.submit, "send"), (send.click, "send"),
                              (reset.click, "reset"), (edit.click, "edit"),
                              (create.click, "approve"), (cancel.click, "cancel")):
            event(
                handle(action), inputs=inputs, outputs=outputs,
                api_visibility="private", concurrency_id="smartdesk",
                concurrency_limit=1, trigger_mode="once",
            )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is missing")
    app = build_app(build_graph(retrieval=args.retrieval, k=args.k))
    app.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
