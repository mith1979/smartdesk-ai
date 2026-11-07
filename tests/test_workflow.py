#!/usr/bin/env python3
"""
File: tests/test_workflow.py

Purpose:
    Run repeatable SmartDesk workflow, failure-handling, and storage checks.
Input:
    Existing scripts/ modules and installed LangGraph dependencies.
    Run from project root: python tests/test_workflow.py
Output:
    unittest results and data/evaluation/workflow_results.json.
FDE connection:
    Weeks 3 and 8: separate control-flow evaluation from model quality.
    Weeks 4-5: exercise actual StateGraph interrupts and session checkpoints.
Side effects:
    Writes the evaluation report and temporary SQLite files (removed on exit).
    All model/API calls are mocked; socket connections are blocked and checked.
    No real credentials are needed or loaded; no GitHub issues are created.
Limitations:
    Does not measure real model routing, answer correctness, or retrieval quality.
    Keep the separate live RAG and conversation checks for those properties.
"""

import io
import json
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import smartdesk_chat as chat
import ticket_store as store
import tickets as api
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.types import Command


EMAIL = "employee@example.com"


def route_decision(intent="kb", context=False, question="Test policy question", reply=""):
    return chat.RouteDecision(
        intent=intent, needs_context=context, question=question, summary="Monitor issue",
        description="A fictional monitor flickers", category="IT", reply=reply,
    )


def live_ticket(number=1):
    return {"number": number, "title": f"Test ticket {number}", "state": "open",
            "url": f"https://example.com/issues/{number}", "comments": [],
            "comments_error": None, "error": None}


class OfflineTest(unittest.TestCase):
    def setUp(self):
        # A swallowed exception cannot hide an unexpected network attempt:
        # tearDown separately verifies the guard was never called.
        self.network = self.enterContext(patch(
            "socket.socket.connect", side_effect=AssertionError("Network blocked in offline tests")
        ))

    def tearDown(self):
        self.network.assert_not_called()


class WorkflowTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.router = Mock()
        self.router.invoke.return_value = route_decision()
        self.router.with_structured_output.return_value = self.router
        self.enterContext(patch.object(chat, "ChatOpenAI", return_value=self.router))
        self.create = self.enterContext(patch.object(chat, "create_ticket"))
        self.lookup = self.enterContext(patch.object(chat, "list_employee_tickets", return_value=[]))
        self.rag = types.ModuleType("rag")
        doc = Document(page_content="Demonstration policy", metadata={
            "chunk_id": "test-001::section-01", "source": "data/kb/test.md", "section": "Test",
        })
        self.rag.retrieve = Mock(return_value=[doc])
        self.rag.generate_answer = Mock(return_value=SimpleNamespace(
            answerable=True, answer="Supported test answer", source_ids=[doc.metadata["chunk_id"]],
        ))
        self.hybrid = types.ModuleType("hybrid_search")
        self.hybrid.retrieve_hybrid = Mock(return_value=[{"document": doc}])
        self.enterContext(patch.dict(sys.modules, {"rag": self.rag, "hybrid_search": self.hybrid}))
        self.graph = chat.build_graph("hybrid")
        self.config = {"configurable": {"thread_id": "offline-test"}}

    def send(self, text, config=None):
        return self.graph.invoke({"messages": [HumanMessage(content=text)]}, config or self.config)

    def resume(self, response):
        return self.graph.invoke(Command(resume=response), self.config)

    def paused_at(self, result, kind):
        self.assertEqual(result["__interrupt__"][0].value["kind"], kind)

    def test_router_failure_preserves_known_email(self):
        self.router.invoke.return_value = route_decision("status")
        self.send(f"Show tickets for {EMAIL}")
        self.router.invoke.side_effect = TimeoutError("Simulated router timeout")
        result = self.send("Check again")
        self.assertEqual(result["email"], EMAIL)
        self.assertIn("could not process", result["messages"][-1].content)
        self.assertNotIn("__interrupt__", result)
        self.create.assert_not_called()
        # A subsequent turn can recover in the same session.
        self.router.invoke.side_effect = None
        result = self.send("Show my tickets again")
        self.assertNotIn("__interrupt__", result)
        self.lookup.assert_called_with(EMAIL)

    def test_retrieval_failure_is_not_missing_evidence(self):
        self.hybrid.retrieve_hybrid.side_effect = TimeoutError("Simulated embedding failure")
        result = self.send("What is the leave policy?")
        self.assertIn("could not complete", result["messages"][-1].content)
        self.assertFalse(result["needs_ticket"])
        self.assertNotIn("__interrupt__", result)
        self.rag.generate_answer.assert_not_called()
        self.create.assert_not_called()

    def test_generation_failure_is_not_a_policy_answer(self):
        self.rag.generate_answer.side_effect = TimeoutError("Simulated model failure")
        result = self.send("What is the leave policy?")
        text = result["messages"][-1].content
        self.assertIn("could not complete", text)
        self.assertNotIn("ANSWER:", text)
        self.assertNotIn("__interrupt__", result)
        self.create.assert_not_called()

    def test_invalid_citation_rejects_generated_answer(self):
        self.rag.generate_answer.return_value = SimpleNamespace(
            answerable=True, answer="UNSUPPORTED RESPONSE", source_ids=["invented-source"],
        )
        result = self.send("Test policy question")
        self.paused_at(result, "offer")
        self.assertIn("could not validate", result["messages"][-1].content)
        self.assertNotIn("UNSUPPORTED RESPONSE", result["messages"][-1].content)
        self.resume("no")
        self.create.assert_not_called()

    def test_empty_retrieval_offers_escalation_without_generation(self):
        self.hybrid.retrieve_hybrid.return_value = []
        result = self.send("A question with no matching passages")
        self.paused_at(result, "offer")
        self.rag.generate_answer.assert_not_called()
        self.resume("no")
        self.create.assert_not_called()

    def test_cancellation_never_calls_ticket_creation(self):
        self.router.invoke.return_value = route_decision("create")
        result = self.send(f"Create a support ticket for {EMAIL}")
        self.paused_at(result, "approval")
        self.create.assert_not_called()
        result = self.resume({"action": "cancel"})
        self.assertIn("Cancelled", result["messages"][-1].content)
        self.create.assert_not_called()

    def test_edit_requires_fresh_approval_and_one_write(self):
        self.router.invoke.return_value = route_decision("create")
        self.send(f"Create a ticket for {EMAIL}")
        revised = {"summary": "Edited title", "description": "Employee-added detail", "category": "IT"}
        result = self.resume({"action": "edit", "draft": revised})
        self.paused_at(result, "approval")
        self.assertEqual(result["__interrupt__"][0].value["draft"], revised)
        self.create.assert_not_called()
        self.create.return_value = {"number": 42, "html_url": "https://example.com/42",
                                    "state": "open", "warning": None}
        result = self.resume({"action": "approve"})
        self.create.assert_called_once_with(EMAIL, revised["summary"], revised["description"],
                                            "IT", confirmed=True)
        self.assertFalse(result["approved"])
        self.assertIn("Created ticket #42", result["messages"][-1].content)

    def test_ticket_creation_failure_does_not_claim_success(self):
        self.router.invoke.return_value = route_decision("create")
        self.create.side_effect = api.TicketError("Creation outcome could not be confirmed; check GitHub before retrying.")
        self.send(f"Create a ticket for {EMAIL}")
        result = self.resume({"action": "approve"})
        self.create.assert_called_once()
        self.assertFalse(result["approved"])
        self.assertNotIn("Created ticket #", result["messages"][-1].content)
        self.assertIn("could not be confirmed", result["messages"][-1].content)

    def test_failed_status_lookup_is_not_reported_as_empty(self):
        self.router.invoke.return_value = route_decision("status")
        self.lookup.side_effect = api.TicketError("Simulated API failure")
        result = self.send(f"Show tickets for {EMAIL}")
        self.assertIn("could not look up", result["messages"][-1].content)
        self.assertNotIn("No locally associated", result["messages"][-1].content)

    def test_multiple_ticket_selection_uses_displayed_choices(self):
        self.router.invoke.return_value = route_decision("status")
        self.lookup.return_value = [live_ticket(1), live_ticket(2)]
        result = self.send(f"Show tickets for {EMAIL}")
        self.paused_at(result, "select")
        result = self.resume(999)
        self.paused_at(result, "select")
        result = self.resume(2)
        self.assertIn("#2: Test ticket 2", result["messages"][-1].content)
        self.create.assert_not_called()

    def test_new_session_does_not_inherit_email(self):
        self.router.invoke.return_value = route_decision("status")
        self.send(f"Show tickets for {EMAIL}")
        result = self.send("Show my tickets")
        self.assertNotIn("__interrupt__", result)
        result = self.send("Show my tickets", {"configurable": {"thread_id": "new-session"}})
        self.paused_at(result, "email")

    def test_resolved_followup_reaches_rag(self):
        original = "I joined in July. How much sick leave do I receive that year?"
        self.send(original)
        self.assertEqual(self.hybrid.retrieve_hybrid.call_args.args[0], original)
        self.router.invoke.return_value = route_decision(
            "clarify", True, "How much sick leave if I joined in September?", "",
        )
        result = self.send("What if I joined in September instead?")
        self.assertEqual(result["intent"], "kb")
        self.assertIn("sick leave", self.hybrid.retrieve_hybrid.call_args.args[0])
        self.assertIn("September", self.hybrid.retrieve_hybrid.call_args.args[0])


class TicketApiTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(api, "load_token", return_value="dummy-offline-token"))
        self.enterContext(patch.object(api, "find_tickets", return_value=[]))
        self.link = self.enterContext(patch.object(api, "link_ticket"))
        self.http = self.enterContext(patch.object(api, "urlopen"))
        self.args = (EMAIL, "Test title", "Fictional details", "IT")

    def test_creation_requires_confirmation(self):
        with self.assertRaises(ValueError):
            api.create_ticket(*self.args)
        self.http.assert_not_called()
        self.link.assert_not_called()

    def test_invalid_mandatory_fields_block_write(self):
        with self.assertRaises(ValueError):
            api.create_ticket(EMAIL, "", "Details", "IT", confirmed=True)
        self.http.assert_not_called()

    def test_write_timeout_warns_about_uncertain_outcome_without_retry(self):
        self.http.side_effect = URLError("Simulated timeout")
        with self.assertRaises(api.TicketError) as caught:
            api.create_ticket(*self.args, confirmed=True)
        self.assertIn("may already exist", str(caught.exception))
        self.http.assert_called_once()
        self.link.assert_not_called()

    def test_http_failure_does_not_save_a_ticket_mapping(self):
        self.http.side_effect = HTTPError("https://example.com", 403, "Forbidden", {}, None)
        with self.assertRaises(api.TicketError):
            api.create_ticket(*self.args, confirmed=True)
        self.http.assert_called_once()
        self.link.assert_not_called()

    def test_mapping_failure_preserves_successful_creation_reference(self):
        self.http.return_value = io.StringIO(json.dumps({
            "number": 42, "html_url": "https://example.com/42", "state": "open",
        }))
        self.link.side_effect = sqlite3.OperationalError("Simulated database failure")
        result = api.create_ticket(*self.args, confirmed=True)
        self.assertEqual(result["number"], 42)
        self.assertFalse(result["mapping_saved"])
        self.assertIn("do not recreate", result["warning"])
        self.http.assert_called_once()

    def test_comment_failure_retains_live_status_and_reports_partial_failure(self):
        with patch.object(api, "find_tickets", return_value=[{"issue_number": 1}]), \
                patch.object(api, "get_ticket", return_value={"title": "Test", "state": "open", "html_url": "test"}), \
                patch.object(api, "get_recent_comments", side_effect=URLError("Simulated outage")):
            result = api.list_employee_tickets(EMAIL)
        self.assertEqual(result[0]["state"], "open")
        self.assertIsNone(result[0]["error"])
        self.assertTrue(result[0]["comments_error"])

    def test_one_failed_ticket_does_not_hide_other_tickets(self):
        with patch.object(api, "find_tickets", return_value=[{"issue_number": 1}, {"issue_number": 2}]), \
                patch.object(api, "get_ticket", side_effect=[URLError("Simulated outage"), {"title": "Second", "state": "closed", "html_url": "test"}]), \
                patch.object(api, "get_recent_comments", return_value=[]):
            result = api.list_employee_tickets(EMAIL)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]["error"])
        self.assertEqual(result[1]["state"], "closed")


class StorageTests(OfflineTest):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "tickets.sqlite3"

    def test_links_persist_and_duplicate_links_do_not_duplicate_tickets(self):
        store.link_ticket(EMAIL, 1, self.database)
        store.link_ticket(EMAIL.upper(), 1, self.database)
        store.link_ticket(EMAIL, 2, self.database)
        self.assertEqual([r["issue_number"] for r in store.find_tickets(EMAIL, self.database)], [2, 1])

    def test_ticket_cannot_be_silently_reassigned(self):
        store.link_ticket(EMAIL, 1, self.database)
        with self.assertRaises(ValueError):
            store.link_ticket("other@example.com", 1, self.database)
        self.assertEqual(store.find_tickets("other@example.com", self.database), [])

    def test_missing_employee_returns_empty_list(self):
        self.assertEqual(store.find_tickets("nobody@example.com", self.database), [])


class RecordingResult(unittest.TextTestResult):
    """Record results without serializing credentials, messages, or tracebacks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def record(self, test, status):
        self.records.append({"test": test.id(), "status": status,
                             "elapsed_seconds": round(time.perf_counter() - self.started, 4)})

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "PASS")

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.record(test, "FAIL")

    def addError(self, test, error):
        super().addError(test, error)
        self.record(test, "ERROR")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    output = PROJECT_DIR / "data" / "evaluation" / "workflow_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline workflow and failure checks; model and API responses mocked",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "successful": result.wasSuccessful(),
        "results": result.records,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
