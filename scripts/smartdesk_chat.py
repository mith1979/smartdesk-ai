#!/usr/bin/env python3
"""
File: scripts/smartdesk_chat.py

Purpose:
    Run SmartDesk's first integrated LangGraph conversation: grounded RAG,
    confirmed GitHub ticket creation, and live status lookup by email.
Input:
    Terminal conversation, existing Chroma index, RAG/ticket helper scripts,
    OPENAI_API_KEY and GitHub settings from the existing .env locations.
Output:
    Grounded answers with sources, ticket previews/results, or live updates.
FDE connection:
    Weeks 2-3: Reuse retrieval/generation and resolve contextual follow-ups.
    Weeks 4-5: Typed state, message reducer, conditional edges, checkpointer,
    interrupt(), and Command(resume=...) for human input.
    Week 7: Reuse external read/write functions as node dependencies.
    These are conceptual connections, not a copy of a class notebook.
Side effects:
    Router/chat, query embedding, and RAG generation use paid model APIs.
    Ticket creation writes to GitHub ONLY after the final CREATE confirmation.
    Successful creation saves the email mapping in local SQLite.
    Status reads GitHub. No automatic retries of ticket writes.
Limitations:
    InMemorySaver keeps conversation state only while this process runs.
    The SQLite ticket mapping persists independently of conversation state.
    Email identifies a demonstration employee; it does not authenticate them.
    Existing RAG citation checks validate IDs, not every claim's grounding.
    A crash during an external write can leave its outcome uncertain.
    Ticket helpers still reside in scripts/ pending the final module refactor.
Failure handling:
    Router and RAG exceptions produce an operational-error response and keep
    session context. They are not classified as missing policy evidence.
    Ticket writes are never automatically retried by this graph.
Usage:
    python scripts/smartdesk_chat.py --retrieval hybrid
    Add --debug to show intent, retrieval question, and retrieved chunk IDs.
    /new starts a separate conversation; /quit exits.
"""

import argparse
import os
import re
import sqlite3
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from verify_github import PROJECT_DIR, FDE_DIR
from ticket_store import normalize_email
from tickets import TicketError, create_ticket, list_employee_tickets, validate_draft


class DeskState(TypedDict, total=False):
    # add_messages appends new messages instead of replacing conversation history.
    messages: Annotated[list[AnyMessage], add_messages]
    email: str
    intent: str
    question: str
    draft: dict[str, str]
    needs_ticket: bool
    wants_ticket: bool
    approved: bool
    candidates: list[dict]
    selected_number: int


class RouteDecision(BaseModel):
    """The model routes and resolves references; Python controls writes."""
    intent: Literal["kb", "create", "status", "chat", "clarify"]
    needs_context: bool = Field(description="True only for an incomplete follow-up requiring earlier conversation; false for a self-contained question")
    question: str = Field(description="Standalone KB question or issue, using conversation context")
    summary: str = Field(description="Short issue title for creation; empty otherwise")
    description: str = Field(description="User-reported issue details; do not invent facts")
    category: Literal["IT", "HR"]
    reply: str = Field(description="Brief reply for chat or clarification only; empty otherwise")


ROUTER_PROMPT = """
You route SmartDesk conversations for fictional NovaTech Solutions.
Choose kb for IT/HR policy or troubleshooting questions; create for an
explicit request to raise a ticket; status for ticket lists/status/updates;
chat for greetings, thanks, and unrelated topics; clarify for ambiguous input.
Use recent conversation to resolve pronouns and follow-up questions into a
standalone question. Retain relevant conditions, without inventing details.
Set needs_context=false for a self-contained question. Do not generalize it.
Preserve dates, months, numbers, negations, and eligibility conditions.
Set needs_context=true only when earlier context is necessary, such as
'What about sick leave?' after a discussion of joining in July.
Needing earlier context does NOT imply clarify. If the conversation resolves
the reference into an IT/HR question, use kb. For example, changing the joining
month after a sick-leave answer is another kb question, not a clarification.
Use clarify only when information needed to identify the question remains
missing AFTER consulting history. Its reply must ask a specific question
about that missing information; never return an empty clarification reply.
Questions about obtaining a replacement or fixing a problem use kb first.
They are not requests to create a support ticket. Earlier ticket discussion
does not make a new troubleshooting question a ticket-creation request.
Use create only when the CURRENT message explicitly requests a support
ticket, for example 'Please create a ticket for my flickering monitor'.
For create, summarize only the reported problem and known conversation facts.
If the issue itself is unclear, clarify rather than invent a ticket.
Infer category IT or HR. Never create, approve, close, or claim to fetch a
ticket yourself. Approval comes from a separate human-confirmation step.
For chat, be brief and explain IT/HR scope when needed; do not answer policies
or troubleshooting from general knowledge. Such questions must use kb.
Do not infer employee emails: Python separately extracts them from user text.
"""


def say(text):
    """Return a new assistant message as a graph state update."""
    return {"messages": [AIMessage(content=text)]}


def build_graph(retrieval="hybrid", k=4, debug=False):
    """Construct one graph and checkpointer for a running CLI session."""
    router = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0, timeout=60, max_retries=2,
    ).with_structured_output(RouteDecision, method="json_schema")

    def route(state):
        try:
            decision = router.invoke([
                SystemMessage(content=ROUTER_PROMPT), *state["messages"][-16:],
            ])
        except Exception as error:
            if debug:
                print(f"[debug] Routing operation failed ({type(error).__name__}).")
            return {
                **say("I could not process your request right now. Please try again. "
                      "No ticket-creation request was sent for this turn."),
                "intent": "chat", "approved": False,
                "needs_ticket": False, "wants_ticket": False,
            }
        latest = str(state["messages"][-1].content)
        intent = decision.intent
        has_prior_user_turn = any(message.type == "human" for message in state["messages"][:-1])
        needs_context = decision.needs_context and has_prior_user_turn
        # Reject the observed inconsistent routing output: the previous turn
        # was a KB question, the model resolved a follow-up, but supplied no
        # clarification question at all. Search the KB with that resolution.
        # A real, nonempty clarification request remains a clarification.
        recovered_follow_up = (
            intent == "clarify" and needs_context and state.get("intent") == "kb"
            and bool(decision.question.strip()) and not decision.reply.strip()
        )
        if recovered_follow_up:
            intent = "kb"
        # Operational questions must visit RAG before ticket drafting.
        # Contextual agreement to the graph's ticket offer is handled by
        # offer_ticket's interrupt, not by this new-message routing path.
        if intent == "create" and not re.search(
            r"\b(?:tickets?|support\s+(?:requests?|cases?))\b", latest, re.IGNORECASE
        ):
            intent = "kb"
        # Self-contained questions go to RAG verbatim. For follow-ups, retain
        # both the original text and the router's resolution to prevent silent
        # loss of dates, amounts, or other user-supplied conditions.
        question = latest
        if needs_context:
            question = (
                f"Employee question: {latest}\n"
                f"Conversation reference resolved as: {decision.question}\n"
                "Preserve all conditions in the employee question; use only KB evidence for policy facts."
            )
        if debug:
            print(f"\n[debug] Model intent: {decision.intent}; effective intent: {intent}")
            print(f"[debug] Needs earlier context: {needs_context}")
            if recovered_follow_up:
                print("[debug] Resolved KB follow-up had no clarification question; routed to RAG.")
            print(f"[debug] Retrieval question: {question}")
        update = {
            "intent": intent, "question": question,
            "approved": False, "wants_ticket": False, "needs_ticket": False,
            "draft": {
                "summary": decision.summary.strip()[:200],
                "description": decision.description.strip(),
                "category": decision.category,
            },
        }
        # Use only email text actually supplied by the employee in this turn.
        matches = re.findall(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", str(latest))
        if len(matches) == 1:
            try:
                update["email"] = normalize_email(matches[0].rstrip(".,;!?"))
            except ValueError:
                pass
        if intent in ("chat", "clarify"):
            fallback = ("Could you clarify which IT or HR question you want help with?"
                        if intent == "clarify" else "How can I help with IT or HR?")
            update.update(say(decision.reply or fallback))
        return update

    def run_rag(state):
        # Lazy imports preserve the existing rag/hybrid module relationship.
        from rag import generate_answer, retrieve
        question = state["question"]
        if retrieval == "hybrid":
            from hybrid_search import retrieve_hybrid
            entries = retrieve_hybrid(question, k=k, fetch_k=max(10, k))
            documents = [entry["document"] for entry in entries]
        else:
            documents = retrieve(question, k)
        if debug:
            print("[debug] Retrieved chunks:")
            for document in documents:
                print(f"  {document.metadata['chunk_id']}")
        if not documents:
            return {**say("I found no KB evidence for that question."), "needs_ticket": True}
        result = generate_answer(question, documents)
        sources = {doc.metadata["chunk_id"]: doc.metadata for doc in documents}
        if (set(result.source_ids) - set(sources)
                or (result.answerable and not result.source_ids)
                or not result.answer.strip()):
            return {**say("I could not validate the generated answer. I can offer escalation."),
                    "needs_ticket": True}
        text = ("ANSWER" if result.answerable else "INSUFFICIENT EVIDENCE")
        text += f":\n{result.answer}"
        if result.source_ids:
            text += "\n\nSources:"
            for chunk_id in dict.fromkeys(result.source_ids):
                metadata = sources[chunk_id]
                text += f"\n  {chunk_id}: {metadata['source']} — {metadata['section']}"
        # The original employee question becomes the initial ticket description.
        draft = {"summary": question[:200], "description": question,
                 "category": state["draft"]["category"]}
        return {**say(text), "needs_ticket": not result.answerable, "draft": draft}

    def answer_kb(state):
        # No interrupts occur in run_rag: catching operational failures here
        # cannot swallow a LangGraph human-input pause. An API error is not
        # evidence that the requested policy is undefined.
        try:
            return run_rag(state)
        except Exception as error:
            if debug:
                print(f"[debug] KB operation failed ({type(error).__name__}).")
            return {
                **say("I could not complete the KB lookup or answer generation. "
                      "Please try again. This does not establish that the policy is missing. "
                      "No ticket-creation request was sent for this turn."),
                "needs_ticket": False, "approved": False,
            }

    def offer_ticket(state):
        # This is only agreement to prepare a draft; final approval comes later.
        response = interrupt({"kind": "offer", "prompt": "Would you like to prepare a support ticket?"})
        wants = isinstance(response, str) and response.strip().lower() in ("yes", "y")
        return {"wants_ticket": wants, **({} if wants else say("No ticket was created."))}

    def collect_email(state):
        if state.get("email"):
            return {}
        response = interrupt({"kind": "email", "prompt": "What employee email should I use?"})
        if isinstance(response, dict) and response.get("cancel") is True:
            return {"email": "", **say("Cancelled. No ticket was created or fetched.")}
        return {"email": normalize_email(str(response))}

    def approve_ticket(state):
        # No writes or model calls precede the interrupt: this node replays on resume.
        draft = state["draft"]
        response = interrupt({"kind": "approval", "email": state["email"], "draft": draft})
        if not isinstance(response, dict):
            return {"approved": False, **say("Ticket creation cancelled.")}
        action = response.get("action")
        if action == "edit":
            revised = response.get("draft", {})
            try:
                email, summary, description, category = validate_draft(
                    state["email"], revised.get("summary", ""),
                    revised.get("description", ""), revised.get("category", ""),
                )
            except ValueError as error:
                return {"approved": False, "wants_ticket": True, **say(str(error))}
            return {"draft": {"summary": summary, "description": description, "category": category},
                    "approved": False, "wants_ticket": True}
        if action == "approve":
            # Revalidate before routing to the node that performs the write.
            validate_draft(state["email"], draft["summary"], draft["description"], draft["category"])
            return {"approved": True, "wants_ticket": False}
        return {"approved": False, "wants_ticket": False, **say("Cancelled. No ticket was created.")}

    def submit_ticket(state):
        if state.get("approved") is not True:
            return say("Ticket was not approved; nothing was submitted.")
        draft = state["draft"]
        try:
            result = create_ticket(state["email"], draft["summary"], draft["description"],
                                   draft["category"], confirmed=True)
        except (TicketError, ValueError, sqlite3.Error, OSError) as error:
            return {"approved": False, **say(f"Ticket operation: {error}")}
        text = f"Created ticket #{result['number']}: {result['html_url']}"
        text += f"\nInitial state: {result['state']}"
        if result.get("warning"):
            text += f"\n{result['warning']}"
        return {"approved": False, **say(text)}

    def load_status(state):
        try:
            candidates = list_employee_tickets(state["email"])
        except (TicketError, ValueError, sqlite3.Error, OSError):
            return {"candidates": [], **say("I could not look up your tickets. Please try again later.")}
        if not candidates:
            return {"candidates": [], **say("No locally associated tickets were found for this email.")}
        return {"candidates": candidates, "selected_number": candidates[0]["number"]}

    def select_ticket(state):
        choices = [{"number": t["number"], "title": t.get("title", "Unavailable"),
                    "state": t.get("state", t.get("error", "Unavailable"))}
                   for t in state["candidates"]]
        response = interrupt({"kind": "select", "tickets": choices})
        if isinstance(response, dict) and response.get("cancel") is True:
            return {"selected_number": 0, **say("Ticket selection cancelled.")}
        selected = int(response)
        if selected not in {ticket["number"] for ticket in state["candidates"]}:
            return {"selected_number": -1, **say("Choose a ticket from the displayed list.")}
        return {"selected_number": selected}

    def show_status(state):
        ticket = next(t for t in state["candidates"] if t["number"] == state["selected_number"])
        if ticket["error"]:
            return say(f"Could not fetch ticket #{ticket['number']}: {ticket['error']}")
        text = f"#{ticket['number']}: {ticket['title']} — {ticket['state']}\n{ticket['url']}"
        if ticket.get("comments_error"):
            text += f"\n{ticket['comments_error']}"
        elif not ticket["comments"]:
            text += "\nNo support comments yet."
        for comment in ticket["comments"]:
            text += f"\n{comment['user']['login']}: {comment['body']}"
        return say(text)

    graph = StateGraph(DeskState)
    for name, node in (("route", route), ("answer_kb", answer_kb),
                       ("offer_ticket", offer_ticket), ("collect_email", collect_email),
                       ("approve_ticket", approve_ticket), ("submit_ticket", submit_ticket),
                       ("load_status", load_status), ("select_ticket", select_ticket),
                       ("show_status", show_status)):
        graph.add_node(name, node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda s: s["intent"], {
        "kb": "answer_kb", "create": "collect_email", "status": "collect_email",
        "chat": END, "clarify": END,
    })
    graph.add_conditional_edges("answer_kb", lambda s: s["needs_ticket"],
                                {True: "offer_ticket", False: END})
    graph.add_conditional_edges("offer_ticket", lambda s: s["wants_ticket"],
                                {True: "collect_email", False: END})
    graph.add_conditional_edges("collect_email", lambda s: (
        "cancel" if not s.get("email") else "status" if s["intent"] == "status" else "create"
    ), {"cancel": END, "status": "load_status", "create": "approve_ticket"})
    graph.add_conditional_edges("approve_ticket", lambda s: (
        "submit" if s.get("approved") else "edit" if s.get("wants_ticket") else "cancel"
    ), {"submit": "submit_ticket", "edit": "approve_ticket", "cancel": END})
    graph.add_edge("submit_ticket", END)
    graph.add_conditional_edges("load_status", lambda s: (
        "empty" if not s["candidates"] else "one" if len(s["candidates"]) == 1 else "many"
    ), {"empty": END, "one": "show_status", "many": "select_ticket"})
    graph.add_conditional_edges("select_ticket", lambda s: (
        "cancel" if s["selected_number"] == 0 else "retry" if s["selected_number"] < 0 else "show"
    ), {"cancel": END, "retry": "select_ticket", "show": "show_status"})
    graph.add_edge("show_status", END)
    return graph.compile(checkpointer=InMemorySaver())


def read_human_input(payload):
    """Translate terminal input into explicit interrupt resume values."""
    kind = payload["kind"]
    if kind == "offer":
        print(f"\nSmartDesk: {payload['prompt']}")
        while True:
            answer = input("You [yes/no]: ").strip().lower()
            if answer in ("yes", "y", "no", "n"):
                return answer
            print("Enter yes or no.")
    if kind == "email":
        while True:
            email = input(f"\nSmartDesk: {payload['prompt']} (/cancel to stop)\nYou: ")
            if email.strip().lower() == "/cancel":
                return {"cancel": True}
            try:
                return normalize_email(email)
            except ValueError as error:
                print(error)
    if kind == "approval":
        draft = payload["draft"]
        print(f"\nTicket preview\nEmail: {payload['email']}")
        for field in ("summary", "description", "category"):
            print(f"{field.title()}: {draft[field]}")
        answer = input("Type CREATE to submit, EDIT to revise, or anything else to cancel: ").strip()
        if answer == "CREATE":
            return {"action": "approve"}
        if answer == "EDIT":
            revised = {field: input(f"New {field} [Enter keeps current]: ").strip() or draft[field]
                       for field in ("summary", "description", "category")}
            return {"action": "edit", "draft": revised}
        return {"action": "cancel"}
    if kind == "select":
        print("\nYour tickets:")
        for ticket in payload["tickets"]:
            print(f"  #{ticket['number']}: {ticket['title']} — {ticket['state']}")
        allowed = {ticket["number"] for ticket in payload["tickets"]}
        while True:
            answer = input("Choose a ticket number, or /cancel: ").strip()
            if answer.lower() == "/cancel":
                return {"cancel": True}
            if answer.isdigit() and int(answer) in allowed:
                return int(answer)
            print("Enter one of the displayed ticket numbers.")
    raise ValueError("Unknown human-input request")


def main():
    parser = argparse.ArgumentParser(description="Converse with SmartDesk")
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--debug", action="store_true", help="Show routing and retrieval diagnostics")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing")
    graph = build_graph(args.retrieval, args.k, debug=args.debug)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 100}
    shown = set()
    print("SmartDesk: Ask about IT/HR, create a ticket, or check ticket status.")
    print("/new starts a new session; /quit exits. Session memory ends on exit.")
    while True:
        try:
            text = input("\nYou: ").strip()
            if text.lower() == "/quit":
                break
            if text.lower() == "/new":
                config["configurable"]["thread_id"] = str(uuid4())
                shown.clear()
                print("New conversation started. Email and chat context reset.")
                continue
            if not text:
                continue
            result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
            while True:
                for message in result.get("messages", []):
                    if message.type == "ai" and message.id not in shown:
                        print(f"\nSmartDesk: {message.content}")
                        shown.add(message.id)
                pending = result.get("__interrupt__", ())
                if not pending:
                    break
                response = read_human_input(pending[0].value)
                result = graph.invoke(Command(resume=response), config=config)
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        except Exception as error:
            # Never blindly resume a failed write node. Start a fresh thread.
            print(f"\nSmartDesk could not complete this turn ({type(error).__name__}).")
            print("Check GitHub before repeating any ticket submission. Conversation reset.")
            config["configurable"]["thread_id"] = str(uuid4())
            shown.clear()


if __name__ == "__main__":
    main()
