import io
import http.client
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError

import llm_connection_test as bench


class WordTokenizer:
    name = "test-words"

    def encode(self, text, **kwargs):
        return text.split()


def sse_event(delta=None, finish_reason=None, **extra):
    return ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {},
             "finish_reason": finish_reason}], **extra}) + "\n\n").encode()


def response_event(event_type, **extra):
    return ("event: " + event_type + "\ndata: " + json.dumps({"type": event_type, **extra}) + "\n\n").encode()


def response_document(text="hello world", *, output=None, status="completed"):
    return {"object": "response", "id": "resp_test", "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "output": output if output is not None else [{"type": "message", "id": "msg_test", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 2}}}


def make_config(**overrides):
    config = bench.BenchmarkConfig("http://127.0.0.1/v1/chat/completions", ("test",),
        "", "", "", 5, 1, True, "system", "user", 256, None, None, None, True, {})
    return replace(config, **overrides)


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []
    paths = []
    auth_headers = []
    variant = "normal"
    parallel_barrier = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(payload)
        self.paths.append(self.path)
        self.auth_headers.append(self.headers.get("Authorization"))
        responses = "input" in payload
        if responses:
            normalized = dict(payload)
            normalized["messages"] = [{"role": item.get("role", "tool"), "content": item.get("content", "")}
                                      for item in payload["input"]]
            if isinstance(payload.get("tool_choice"), dict):
                normalized["tool_choice"] = {"type": "function", "function": {"name": payload["tool_choice"]["name"]}}
            payload = normalized
        variant = self.variant
        if variant == "responses_http_error":
            variant = "http_error" if responses else "normal"
        if variant == "managed_forbidden":
            variant = "forbidden" if self.headers.get("Authorization") == "Bearer managed-token" else "normal"
        if variant == "split_identity_protocol":
            unauthorized = (responses and self.headers.get("Authorization") == "Bearer managed-token"
                            or not responses and self.headers.get("Authorization") == "Bearer app-token")
            variant = "forbidden" if unauthorized else "normal"
        if variant.startswith("child_"):
            child_answer = payload.get("tool_choice") == "none" and any(
                message["role"] == "user" and "Topic: latency" in message.get("content", "")
                for message in payload["messages"])
            variant = variant.removeprefix("child_") if child_answer else "normal"
        if variant in {"http_error", "unauthorized", "forbidden", "permission_detail"}:
            self.send_response({"http_error": 400, "unauthorized": 401, "forbidden": 403, "permission_detail": 401}[variant])
            body = json.dumps({"error": {"code": "PermissionDenied",
                "message": "Principal lacks inference access. private-client-secret fake-azure-access-token"}}).encode() if variant == "permission_detail" else b""
            self.send_header("Content-Length", str(len(body)))
            self.send_header("apim-request-id", "synthetic-request-id")
            self.end_headers()
            self.wfile.write(body)
            return
        message = {"role": "assistant", "content": "hello world"}
        finish_reason = "stop"
        if variant == "long":
            message["content"] = " ".join(["token"] * 1201)
        elif variant == "oversized_short":
            message["content"] = " ".join(["token"] * 200)
        elif variant == "truncated":
            finish_reason = "length"
        if isinstance(payload.get("tool_choice"), dict) and variant != "ignore_tools":
            name = payload["tool_choice"]["function"]["name"]
            if name == "delegate":
                arguments = {"topics": ["latency", "throughput"]}
            else:
                arguments = {"topic": "throughput" if "Topic: throughput" in str(payload["messages"]) else "latency"}
                if self.parallel_barrier is not None:
                    try:
                        self.parallel_barrier.wait(timeout=3)
                    except threading.BrokenBarrierError:
                        self.send_response(500)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_test",
                "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            finish_reason = "tool_calls"
            if variant == "malformed_tool":
                message["tool_calls"][0]["function"]["arguments"] = "not json"
        if responses:
            reasoning = {"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "cipher-test"}
            if message.get("tool_calls"):
                call = message["tool_calls"][0]
                item = {"type": "function_call", "id": "fc_item", "call_id": call["id"],
                        "name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                document = response_document(output=[reasoning, item])
            else:
                document = response_document(message["content"], status="incomplete" if finish_reason == "length" else "completed")
                document["output"].insert(0, reasoning)
            if payload["stream"]:
                body = response_event("response.created", response={"status": "in_progress"})
                body += response_event("response.output_item.done", output_index=0, item=reasoning)
                if message.get("tool_calls"):
                    body += response_event("response.output_item.added", output_index=1, item={**item, "arguments": ""})
                    body += response_event("response.function_call_arguments.delta", output_index=1, delta=item["arguments"])
                    body += response_event("response.output_item.done", output_index=1, item=item)
                else:
                    text = message["content"]
                    midpoint = max(1, len(text) // 2)
                    body += response_event("response.output_text.delta", output_index=1, content_index=0, delta=text[:midpoint])
                    body += response_event("response.output_text.delta", output_index=1, content_index=0, delta=text[midpoint:])
                if variant != "interrupted":
                    body += response_event("response." + document["status"], response=document)
                if variant == "token_error":
                    body = response_event("error", message="fake-azure-access-token private-client-secret")
            else:
                body = json.dumps(document).encode()
            content_type = "text/event-stream" if payload["stream"] else "application/json"
        elif payload["stream"]:
            if message.get("tool_calls"):
                delta = {"tool_calls": [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]}
                body = sse_event(delta)
            else:
                text = message["content"]
                midpoint = max(1, len(text) // 2)
                body = sse_event({"content": text[:midpoint]}) + sse_event({"content": text[midpoint:]})
            body += sse_event(finish_reason=finish_reason) + b"data: [DONE]\n\n"
            if variant == "interrupted":
                body = body.replace(b"data: [DONE]\n\n", b"")
            if variant == "token_error":
                body = b'data: {"error": {"message": "fake-azure-access-token private-client-secret"}}\n\n'
            content_type = "text/event-stream"
        else:
            body = json.dumps({"choices": [{"message": message, "finish_reason": finish_reason}]}).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body) + (100 if variant == "incomplete_body" else 0)))
        if variant == "incomplete_body":
            self.send_header("Connection", "close")
            self.close_connection = True
        self.send_header("Server-Timing", "prefill;dur=12")
        self.end_headers()
        self.wfile.write(body)


class ProtocolTests(unittest.TestCase):
    def test_responses_stream_text_timing_excludes_lifecycle_and_completion(self):
        body = (response_event("response.created", response={"status": "in_progress"})
            + response_event("response.output_text.delta", delta="hello ")
            + response_event("response.output_text.delta", delta="world")
            + response_event("response.completed", response=response_document()))
        timing = bench.Timing(0, request_sent_at=1, completed_at=20)
        with patch.object(bench.time, "perf_counter", side_effect=[1.1, 1.2, 2, 4]):
            parsed = bench.consume_sse(io.BytesIO(body), timing, protocol="responses")
        result = bench.build_call_result("test", 1, True, timing, parsed, WordTokenizer(), 200)
        self.assertEqual(result["metrics"]["ttft_ms"], 2000)
        self.assertEqual(result["metrics"]["ttlt_ms"], 4000)
        self.assertEqual(result["metrics"]["tps"], 0.5)
        self.assertEqual(result["tokens"]["api_completion_tokens"], 5)
        self.assertEqual(result["tokens"]["visible_output_tokens"], 2)
        self.assertEqual(result["tokens"]["reasoning_tokens"], 2)
        self.assertEqual(result["tokens"]["cached_input_tokens"], 3)

    def test_responses_json_has_no_streaming_metrics(self):
        parsed = bench.consume_json(io.BytesIO(json.dumps(response_document()).encode()), protocol="responses")
        result = bench.build_call_result("test", 1, False, bench.Timing(0, completed_at=1), parsed, WordTokenizer(), 200)
        for name in ("ttft_ms", "ttlt_ms", "tps", "prefill_proxy_ms"):
            self.assertIsNone(result["metrics"][name])
        self.assertEqual(result["api_usage"]["input_tokens"], 10)

    def test_responses_stream_assembles_tools_and_preserves_reasoning_history(self):
        reasoning = {"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "complete-cipher"}
        call = {"type": "function_call", "id": "fc_item", "call_id": "call_test", "name": "lookup_fact", "arguments": '{"topic":"latency"}'}
        body = (response_event("response.output_item.added", output_index=0, item={**reasoning, "encrypted_content": "partial"})
            + response_event("response.output_item.done", output_index=0, item=reasoning)
            + response_event("response.output_item.added", output_index=1, item={**call, "arguments": ""})
            + response_event("response.function_call_arguments.delta", output_index=1, delta='{"topic":')
            + response_event("response.function_call_arguments.delta", output_index=1, delta='"latency"}')
            + response_event("response.output_item.done", output_index=1, item=call)
            + response_event("response.completed", response=response_document(output=[reasoning, call])))
        parsed = bench.consume_sse(io.BytesIO(body), bench.Timing(0), protocol="responses")
        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.text_chunk_count, 0)
        self.assertEqual(parsed.finish_reason, "tool_calls")
        self.assertEqual(parsed.tool_calls[0]["id"], "call_test")
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"], call["arguments"])
        self.assertEqual(parsed.response_output[0]["encrypted_content"], "complete-cipher")

    def test_responses_missing_terminal_retains_partial_text_and_tools(self):
        body = (response_event("response.output_text.delta", delta="partial")
            + response_event("response.output_item.added", output_index=0,
                item={"type": "function_call", "call_id": "call_test", "name": "lookup_fact", "arguments": ""})
            + response_event("response.function_call_arguments.delta", output_index=0, delta='{"topic":'))
        partial = bench.StreamResult("", None, 0)
        with self.assertRaisesRegex(bench.BenchmarkError, "Incomplete Responses"):
            bench.consume_sse(io.BytesIO(body), bench.Timing(0), partial, protocol="responses")
        self.assertEqual(partial.text, "partial")
        self.assertEqual(partial.tool_calls[0]["function"]["arguments"], '{"topic":')

    def test_responses_truncation_error_and_malformed_shapes(self):
        parsed = bench.parse_response_document(response_document(status="incomplete"))
        self.assertEqual(parsed.finish_reason, "length")
        for document in ({}, {**response_document(), "status": "in_progress"},
                         {**response_document(), "error": {"message": "synthetic failure"}},
                         {**response_document(), "output": "bad"}):
            with self.subTest(document=document), self.assertRaises(bench.BenchmarkError):
                bench.parse_response_document(document)
        for body in (b"data: [DONE]\n\n", response_event("response.output_text.delta", delta=123),
                     response_event("response.completed", response=response_document()),
                     response_event("error", message="synthetic failure")):
            with self.subTest(body=body), self.assertRaises(bench.BenchmarkError):
                bench.consume_sse(io.BytesIO(body), bench.Timing(0), protocol="responses")

    def test_base_and_full_urls_resolve_for_both_protocols(self):
        for path in ("/openai/v1", "/openai/v1/", "/openai/v1/chat/completions", "/openai/v1/responses/"):
            for protocol, suffix in bench._PROTOCOL_PATHS.items():
                with self.subTest(path=path, protocol=protocol):
                    endpoint = bench.parse_endpoint("https://example.com" + path + "?api-version=test")
                    resolved = bench.resolve_protocol_endpoint(endpoint, protocol)
                    self.assertEqual(resolved.path, "/openai/v1" + suffix)
                    self.assertEqual(resolved.query, "api-version=test")
                    self.assertNotIn("?", bench.display_endpoint(resolved))

    def test_responses_payload_maps_budget_reasoning_and_tool_choice(self):
        for streaming in (True, False):
            config = make_config(protocol="responses", max_tokens=None, max_completion_tokens=123,
                                 reasoning_effort="low", temperature=0.1)
            body, _ = bench.build_request(config, "test", streaming=streaming,
                tools=[bench.agent_tool(False)], tool_choice={"type": "function", "function": {"name": "lookup_fact"}})
            payload = json.loads(body)
            self.assertEqual(payload["input"], [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}])
            self.assertEqual(payload["max_output_tokens"], 123)
            self.assertEqual(payload["reasoning"], {"effort": "low"})
            self.assertEqual(payload["stream"], streaming)
            self.assertFalse(payload["store"])
            self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
            self.assertEqual(payload["tool_choice"], {"type": "function", "name": "lookup_fact"})
            self.assertEqual(payload["tools"][0]["name"], "lookup_fact")
            for name in ("messages", "max_tokens", "max_completion_tokens", "reasoning_effort", "stream_options"):
                self.assertNotIn(name, payload)

    def test_responses_input_preserves_raw_reasoning_items_and_call_ids(self):
        output = [{"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "encrypted"},
                  {"type": "function_call", "id": "fc_item", "call_id": "call_test", "name": "lookup_fact", "arguments": "{}"}]
        messages = [{"role": "assistant", "content": "", "tool_calls": [], "responses_output": output},
                    {"role": "tool", "tool_call_id": "call_test", "content": "local result"}]
        items = bench.responses_input(messages)
        self.assertEqual(items[:2], output)
        self.assertEqual(items[2], {"type": "function_call_output", "call_id": "call_test", "output": "local result"})


class MeasurementTests(unittest.TestCase):
    def test_endpoint_diagnostics_are_bounded_and_never_dump_non_json(self):
        response = Mock(status=401, length=None)
        response.getheader.return_value = None
        response.read1.return_value = b"private data " * 10000
        reader = bench.DeadlineResponse(response, None, float("inf"))
        details = bench.endpoint_error_details(response, reader, str)
        self.assertEqual(details["http_status"], 401)
        self.assertNotIn("private data", json.dumps(details))
        response.read1.assert_called_once()

    def test_chunked_and_close_delimited_response_framing(self):
        for headers, body, incomplete in (
            (b"Transfer-Encoding: chunked\r\n", b"5\r\nhello\r\n0\r\n\r\n", False),
            (b"Transfer-Encoding: chunked\r\n", b"5\r\nhello\r\n", True),
            (b"Connection: close\r\n", b"hello", False),
            (b"Content-Length: 5\r\n", b"hello", False),
            (b"Content-Length: 6\r\n", b"hello", True),
        ):
            with self.subTest(headers=headers, incomplete=incomplete):
                sock = Mock()
                sock.makefile.return_value = io.BytesIO(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body)
                response = http.client.HTTPResponse(sock)
                response.begin()
                reader = bench.DeadlineResponse(response, None, float("inf"))
                if incomplete:
                    with self.assertRaises(http.client.IncompleteRead):
                        reader.read()
                else:
                    self.assertEqual(reader.read(), b"hello")

    def test_decode_excludes_first_chunk_and_stream_tail(self):
        timing = bench.Timing(0, request_sent_at=1, first_token_at=2,
                              last_token_at=4, completed_at=10)
        stream = bench.StreamResult("one two three four", None, 5,
                                    first_chunk="one two", text_chunk_count=2)
        result = bench.build_call_result("test", 1, True, timing, stream, WordTokenizer(), 200)
        self.assertEqual(result["metrics"]["tps"], 1)
        self.assertEqual(result["metrics"]["ttft_ms"], 2000)
        self.assertEqual(result["metrics"]["ttlt_ms"], 4000)
        self.assertEqual(result["metrics"]["prefill_proxy_ms"], 1000)
        self.assertEqual(result["metrics"]["end_to_end_output_tokens_per_second"], 0.4)

    def test_single_chunk_has_no_decode_tps(self):
        result = bench.build_call_result("test", 1, True,
            bench.Timing(0, first_token_at=1, last_token_at=1, completed_at=2),
            bench.StreamResult("hello", None, 1, first_chunk="hello", text_chunk_count=1),
            WordTokenizer(), 200)
        self.assertIsNone(result["metrics"]["tps"])

    def test_non_streaming_unobservable_metrics_are_null(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "hello"},
            "finish_reason": "stop"}], "usage": {"completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 29}}}).encode())
        stream = bench.consume_json(response)
        result = bench.build_call_result("test", 1, False, bench.Timing(0, completed_at=2),
                                          stream, WordTokenizer(), 200)
        for name in ("ttft_ms", "ttlt_ms", "prefill_proxy_ms", "tps"):
            self.assertIsNone(result["metrics"][name])
        self.assertEqual(result["tokens"]["output_tokens"], 1)
        self.assertEqual(result["tokens"]["reasoning_tokens"], 29)

    def test_sse_tracks_text_not_usage_or_tool_chunks(self):
        body = (sse_event({"role": "assistant"}) + sse_event({"content": "hello "})
                + sse_event({"content": "world"}) + sse_event(finish_reason="stop")
                + b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n\n'
                + b"data: [DONE]\n\n")
        timing = bench.Timing(0)
        with patch.object(bench.time, "perf_counter", side_effect=range(1, 20)):
            stream = bench.consume_sse(io.BytesIO(body), timing)
        self.assertEqual(stream.text, "hello world")
        self.assertEqual(stream.text_chunk_count, 2)
        self.assertLess(timing.first_token_at, timing.last_token_at)
        self.assertEqual(stream.first_chunk, "hello ")

    def test_incomplete_stream_is_failure(self):
        partial = bench.StreamResult("", None, 0)
        timing = bench.Timing(0)
        with self.assertRaisesRegex(bench.BenchmarkError, "Incomplete"):
            bench.consume_sse(io.BytesIO(sse_event({"content": "partial"})), timing, partial)
        self.assertEqual(partial.text, "partial")
        self.assertEqual(partial.event_count, 1)
        self.assertIsNotNone(timing.first_token_at)

    def test_read_failure_preserves_received_usage_and_text(self):
        body = (sse_event({"content": "received text"})
                + b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n\n')

        class InterruptedReader(io.BytesIO):
            def readline(self):
                line = super().readline()
                if not line:
                    raise TimeoutError("synthetic read timeout")
                return line

        partial = bench.StreamResult("", None, 0)
        with self.assertRaises(TimeoutError):
            bench.consume_sse(InterruptedReader(body), bench.Timing(0), partial)
        self.assertEqual(partial.text, "received text")
        self.assertEqual(partial.usage, {"completion_tokens": 3})

    def test_tool_fragments_are_assembled(self):
        body = (sse_event({"tool_calls": [{"index": 0, "id": "call_1", "function": {
            "name": "delegate", "arguments": '{"task":'}}]})
            + sse_event({"tool_calls": [{"index": 0, "function": {"arguments": '"test"}'}}]})
            + sse_event(finish_reason="tool_calls") + b"data: [DONE]\n\n")
        stream = bench.consume_sse(io.BytesIO(body), bench.Timing(0))
        self.assertEqual(json.loads(stream.tool_calls[0]["function"]["arguments"]), {"task": "test"})
        self.assertIsNone(bench.Timing(0).first_token_at)

    def test_real_tokenizer_api_with_offline_encoding(self):
        tokenizer = bench.tiktoken.Encoding(name="local-bytes", pat_str=r"(?s).",
            mergeable_ranks={bytes([value]): value for value in range(256)}, special_tokens={})
        self.assertEqual(bench.count_visible_tokens("test", tokenizer), (4, "tiktoken:local-bytes"))

    def test_deadline_checked_between_partial_line_reads(self):
        class PartialReader:
            length = None

            def read1(self, amount):
                return b"data: "
        reader = bench.DeadlineResponse(PartialReader(), None, 2)
        with patch.object(bench.time, "perf_counter", side_effect=[1, 3]):
            with self.assertRaisesRegex(bench.BenchmarkError, "deadline"):
                reader.readline()

    def test_malformed_sse_and_json_fail_cleanly(self):
        for body in (b"data: not-json\n\n", b"data: []\n\n", b"data: \xff\n\n",
                     sse_event({"tool_calls": 1}),
                     sse_event({"tool_calls": [{"index": 0, "function": True}]})):
            with self.subTest(body=body), self.assertRaises(bench.BenchmarkError):
                bench.consume_sse(io.BytesIO(body), bench.Timing(0))
        for document in ({"choices": []}, {"choices": [{"message": {}, "finish_reason": None}]},
                         {"choices": [{"message": {"tool_calls": {}}, "finish_reason": "stop"}], "error": "failed"}):
            with self.assertRaises(bench.BenchmarkError):
                bench.consume_json(io.BytesIO(json.dumps(document).encode()))


class ConnectionDeadlineTests(unittest.TestCase):
    def connect_with_clock(self, durations, timeout=5, dns_duration=0):
        clock = [0.0]
        sockets = []
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", index + 1))
                     for index in range(len(durations))]

        class FakeSocket:
            def __init__(self, *args):
                self.index = len(sockets)
                self.timeouts = []
                self.closed = False
                sockets.append(self)

            def settimeout(self, value):
                self.timeouts.append(value)

            def connect(self, address):
                duration, succeeds = durations[self.index]
                clock[0] += duration
                if not succeeds:
                    raise TimeoutError("simulated connection timeout")

            def close(self):
                self.closed = True

        def resolve(*args, **kwargs):
            clock[0] += dns_duration
            return addresses

        timing = bench.Timing(0)
        failure = None
        with patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(bench.socket, "getaddrinfo", side_effect=resolve), \
             patch.object(bench.socket, "socket", FakeSocket):
            try:
                bench.open_timed_tcp_socket("test.invalid", 443, timeout, timing)
            except bench.BenchmarkError as error:
                failure = error
        return timing, sockets, failure

    def test_failed_attempts_share_budget_and_count_toward_tcp_time(self):
        timing, sockets, failure = self.connect_with_clock([(2, False), (1, True)], dns_duration=1)
        self.assertIsNone(failure)
        self.assertEqual([candidate.timeouts[0] for candidate in sockets], [4, 2])
        self.assertEqual(timing.dns_seconds, 1)
        self.assertEqual(timing.tcp_seconds, 3)
        self.assertEqual(timing.tcp_attempts, 2)
        self.assertTrue(sockets[0].closed)

    def test_no_new_attempt_after_deadline_exhausted(self):
        timing, sockets, failure = self.connect_with_clock([(5, False), (1, True)])
        self.assertIsInstance(failure, bench.BenchmarkError)
        self.assertEqual(len(sockets), 1)
        self.assertTrue(sockets[0].closed)
        self.assertEqual(timing.tcp_seconds, 5)

    def test_dns_overrun_prevents_tcp_attempt(self):
        timing, sockets, failure = self.connect_with_clock([(1, True)], dns_duration=6)
        self.assertIsInstance(failure, bench.BenchmarkError)
        self.assertEqual(sockets, [])
        self.assertEqual(timing.dns_seconds, 6)

    def test_late_success_is_closed_and_rejected(self):
        _, sockets, failure = self.connect_with_clock([(6, True)])
        self.assertIsInstance(failure, bench.BenchmarkError)
        self.assertTrue(sockets[0].closed)

    def test_tls_uses_only_remaining_request_budget(self):
        clock = [4.0]
        timing = bench.Timing(0, deadline=5)
        raw_socket = Mock()
        tls_socket = Mock()
        context = Mock()

        def handshake(*args, **kwargs):
            clock[0] += 0.5
            return tls_socket

        context.wrap_socket.side_effect = handshake
        with patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(bench, "open_timed_tcp_socket", return_value=raw_socket) as tcp:
            connection = bench.TimedHTTPSConnection("test.invalid", 443, 5, timing)
            connection._context = context
            connection.connect()
            connection.close()
        self.assertEqual(tcp.call_args.kwargs["deadline"], 5)
        raw_socket.settimeout.assert_called_once_with(1)
        tls_socket.settimeout.assert_called_once_with(0.5)
        self.assertEqual(timing.tls_seconds, 0.5)


class TransportTests(unittest.TestCase):
    def test_live_progress_covers_protocol_short_sequence_first_text_and_parallel_children(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both",
                             scenarios=("short_burst", "agent_parallel"), short_requests=2)
        output = io.StringIO()
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             bench.ProgressReporter(stream=output) as reporter:
            result = bench.run_benchmarks(config)
            self.assertFalse(reporter.active)
        self.assertTrue(result["success"])
        text = output.getvalue()
        for expected in ("Responses endpoint testing - model=test - short output", "Chat Completions endpoint testing",
                         "measured round 1/1", "request 2/2", "authenticating", "waiting for response headers",
                         "first visible text received; generating", "delegating parallel subagents",
                         "agent=subagent:latency - step 1/4", "agent=subagent:throughput - step 1/4", "[PASS"):
            self.assertIn(expected, text)
        self.assertNotIn(config.user_prompt, text)
        self.assertFalse(reporter.thread.is_alive())

    def test_failed_access_validation_progress_is_immediate_and_safe(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1?secret=query-value", protocol="both",
                             api_key="fake-key", auth_header="Authorization", auth_scheme="Bearer")
        output = io.StringIO()
        with patch.object(MockHandler, "variant", "http_error"), bench.ProgressReporter(stream=output) as reporter:
            result = bench.run_validation(config)
            self.assertFalse(reporter.active)
        self.assertFalse(result["success"])
        self.assertIn("access validation", output.getvalue())
        self.assertIn("[FAILED", output.getvalue())
        self.assertIn("HTTP 400", output.getvalue())
        for secret in ("fake-key", "query-value", "Authorization"):
            self.assertNotIn(secret, output.getvalue())

    def test_validate_auto_lists_all_methods_in_order_after_one_success(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both", auth_type="auto",
            azure_tenant_id="tenant", azure_client_id="app", azure_client_secret="fake-secret")
        managed = Mock(spec=["get_token", "close"])
        managed.get_token.side_effect = bench.CredentialUnavailableError("managed identity unavailable")
        app = Mock(spec=["get_token", "close"])
        app.get_token.return_value = AccessToken("app-token", 9999999999)
        requests, saved, order = [], [], []

        def managed_credential(**kwargs):
            order.append("managed")
            return managed

        def app_credential(**kwargs):
            order.append("app")
            return app

        with patch.object(bench, "ManagedIdentityCredential", side_effect=managed_credential), \
             patch.object(bench, "ClientSecretCredential", side_effect=app_credential), \
             patch.object(MockHandler, "requests", requests):
            result = bench.run_validation(config, on_record=saved.append)
        self.assertTrue(result["success"])
        self.assertEqual(order, ["managed", "app"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(result["maximum_model_requests"], 6)
        self.assertEqual(saved, result["probes"])
        self.assertEqual([method["auth_type"] for method in result["auth_results"]], list(bench._AUTH_CHAIN))
        self.assertEqual([method["status"] for method in result["auth_results"]], ["failed", "success", "failed"])
        self.assertEqual(result["usable_auth_types"], ["azure_app_registration"])
        output = io.StringIO()
        with patch.object(sys, "stdout", output):
            bench.print_validation_result(result)
        self.assertIn("Managed Identity - Failed", output.getvalue())
        self.assertIn("App Registration - Success", output.getvalue())
        self.assertIn("LLM API Key - Failed", output.getvalue())
        markdown = bench.render_markdown(result)
        self.assertIn("# LLM Access Validation", markdown)
        self.assertIn("## Authentication Checks", markdown)
        self.assertIn("| azure\\_managed\\_identity | False | Failed |", markdown)
        self.assertIn("| azure\\_app\\_registration | True | Success |", markdown)
        self.assertNotIn("fake-secret", markdown)
        managed.close.assert_called_once()
        app.close.assert_called_once()

    def test_validate_auto_checks_app_and_key_even_when_managed_identity_succeeds(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both", auth_type="auto",
            api_key="fake-key", auth_header="Authorization", auth_scheme="Bearer", azure_tenant_id="tenant",
            azure_client_id="app-client", azure_managed_identity_client_id="mi-client", azure_client_secret="fake-secret")
        managed, app = Mock(spec=["get_token", "close"]), Mock(spec=["get_token", "close"])
        managed.get_token.return_value = AccessToken("managed-token", 9999999999)
        app.get_token.return_value = AccessToken("app-token", 9999999999)
        headers = []
        with patch.object(bench, "ManagedIdentityCredential", return_value=managed) as managed_constructor, \
             patch.object(bench, "ClientSecretCredential", return_value=app) as app_constructor, \
             patch.object(MockHandler, "auth_headers", headers):
            result = bench.run_validation(config)
        self.assertTrue(result["success"])
        self.assertTrue(all(method["success"] for method in result["auth_results"]))
        self.assertEqual(headers, ["Bearer managed-token"] * 2 + ["Bearer app-token"] * 2 + ["Bearer fake-key"] * 2)
        self.assertEqual(managed_constructor.call_args.kwargs["client_id"], "mi-client")
        self.assertEqual(app_constructor.call_args.kwargs["client_id"], "app-client")
        managed.get_token.assert_called_once()
        app.get_token.assert_called_once()

    def test_validate_auto_reports_total_failure_without_sending_unconfigured_credentials(self):
        config = make_config(auth_type="auto", protocol="both", azure_managed_identity_client_id=None)
        with patch.object(bench, "ManagedIdentityCredential") as managed, \
             patch.object(bench, "ClientSecretCredential") as app, patch.object(bench, "create_connection") as connection:
            result = bench.run_validation(config)
        self.assertFalse(result["success"])
        self.assertEqual(len(result["auth_results"]), 3)
        self.assertTrue(all(method["status"] == "failed" for method in result["auth_results"]))
        self.assertEqual(result["usable_auth_types"], [])
        managed.assert_not_called()
        app.assert_not_called()
        connection.assert_not_called()

    def test_validate_auto_success_requires_endpoint_access_not_just_a_token(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both", auth_type="auto",
            azure_tenant_id="tenant", azure_client_id="app", azure_client_secret="fake-secret")
        for variant, overall_success in (("managed_forbidden", True), ("split_identity_protocol", False)):
            with self.subTest(variant=variant):
                managed, app = Mock(spec=["get_token", "close"]), Mock(spec=["get_token", "close"])
                managed.get_token.return_value = AccessToken("managed-token", 9999999999)
                app.get_token.return_value = AccessToken("app-token", 9999999999)
                with patch.object(bench, "ManagedIdentityCredential", return_value=managed), \
                     patch.object(bench, "ClientSecretCredential", return_value=app), \
                     patch.object(MockHandler, "variant", variant):
                    result = bench.run_validation(config)
                self.assertEqual(result["success"], overall_success)
                self.assertTrue(result["auth_results"][0]["authentication_success"])
                self.assertFalse(result["auth_results"][0]["success"])
                self.assertEqual(result["auth_results"][1]["success"], overall_success)
                self.assertEqual(result["usable_auth_types"], ["azure_app_registration"] if overall_success else [])
                managed.close.assert_called_once()
                app.close.assert_called_once()

    def test_validate_api_key_rejection_does_not_suggest_app_role_assignment(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", api_key="fake-key",
                             auth_header="api-key", auth_scheme="")
        with patch.object(MockHandler, "variant", "unauthorized"):
            result = bench.run_validation(config)
        self.assertFalse(result["success"])
        self.assertIn("API key", result["probes"][0]["guidance"])
        self.assertNotIn("service principal", result["probes"][0]["guidance"])

    def test_validate_checks_first_model_once_per_protocol_with_no_tokenizer_or_benchmark(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/openai/v1", protocol="both", mode="both",
            models=("first", "second"), rounds=10, warmup_rounds=3, scenarios=("agent_parallel",), max_tokens=256,
            auth_type="azure_app_registration", azure_client_id="app", azure_tenant_id="tenant", azure_client_secret="fake-secret")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-token", 9999999999)
        requests = []
        saved = []
        with patch.object(bench, "ClientSecretCredential", return_value=credential), \
             patch.object(bench, "ManagedIdentityCredential") as managed, \
             patch.object(bench, "resolve_tokenizer") as tokenizer, \
             patch.object(MockHandler, "requests", requests), patch.object(bench, "run_agent_scenario") as agent:
            result = bench.run_validation(config, on_record=saved.append)
        self.assertTrue(result["success"], result)
        self.assertEqual(len(requests), 2)
        self.assertEqual(saved, result["probes"])
        self.assertTrue(all(request["model"] == "first" and not request["stream"] for request in requests))
        self.assertEqual(requests[0]["max_tokens"], 64)
        self.assertEqual(requests[1]["max_output_tokens"], 64)
        tokenizer.assert_not_called()
        managed.assert_not_called()
        agent.assert_not_called()
        credential.get_token.assert_called_once()
        credential.close.assert_called_once()

    def test_validate_distinguishes_identity_failure_from_endpoint_permission_denied(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both",
            auth_type="azure_app_registration", azure_tenant_id="tenant", azure_client_id="app", azure_client_secret="private-client-secret")
        for credential_available in (True, False):
            with self.subTest(credential_available=credential_available):
                credential = Mock(spec=["get_token", "close"])
                if credential_available:
                    credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
                else:
                    credential.get_token.side_effect = ClientAuthenticationError("AADSTS7000215: private-client-secret")
                requests = []
                with patch.object(bench, "ClientSecretCredential", return_value=credential), \
                     patch.object(MockHandler, "variant", "permission_detail"), patch.object(MockHandler, "requests", requests):
                    result = bench.run_validation(config)
                self.assertFalse(result["success"])
                self.assertEqual(len(requests), 2 if credential_available else 0)
                probe = result["probes"][0]
                self.assertEqual(probe["stage"], "endpoint" if credential_available else "authentication")
                self.assertEqual(probe["authentication_success"], credential_available)
                if credential_available:
                    self.assertEqual(probe["endpoint_error"]["code"], "PermissionDenied")
                    self.assertIn("service principal", probe["guidance"])
                else:
                    self.assertEqual(result["skipped_protocols"], ["responses"])
                output = io.StringIO()
                with patch.object(sys, "stdout", output):
                    bench.print_validation_result(result)
                self.assertIn("Credential acquisition: PASS" if credential_available else "Credential acquisition: FAIL", output.getvalue())
                if not credential_available:
                    self.assertIn("AADSTS7000215", output.getvalue())
                    self.assertIn("AADSTS7000215", json.dumps(result))
                for secret in ("private-client-secret", "fake-azure-access-token"):
                    self.assertNotIn(secret, output.getvalue())
                    self.assertNotIn(secret, json.dumps(result))

    def test_validate_truncated_small_output_proves_access_not_normal_completion(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both", max_tokens=16)
        with patch.object(MockHandler, "variant", "truncated"):
            result = bench.run_validation(config)
        self.assertTrue(result["success"])
        self.assertEqual(result["token_budget_per_request"], 16)
        self.assertTrue(all(probe["finish_reason"] == "length" and probe["warning"] for probe in result["probes"]))

    def test_validate_connection_setup_failure_is_not_a_credential_failure(self):
        config = make_config(auth_type="azure_app_registration", azure_tenant_id="tenant",
                             azure_client_id="app", azure_client_secret="fake-secret")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-token", 9999999999)
        with patch.object(bench, "ClientSecretCredential", return_value=credential), \
             patch.object(bench, "create_connection", side_effect=OSError("synthetic network error")):
            result = bench.run_validation(config)
        self.assertFalse(result["success"])
        self.assertEqual(result["probes"][0]["stage"], "network")
        self.assertTrue(result["probes"][0]["authentication_success"])
        self.assertIsNone(result["probes"][0]["http_status"])
        credential.get_token.assert_called_once()
        credential.close.assert_called_once()

    def test_cli_validate_saves_json_and_jsonl_without_running_benchmark_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            report = Path(directory) / "validation.json"
            jsonl = Path(directory) / "validation.jsonl"
            markdown_path = Path(directory) / "validation.md"
            config_path.write_text(f'[llm]\nendpoint="http://127.0.0.1:{self.server.server_port}/v1"\n'
                'models=["first","second"]\nauth_header=""\n[request]\nuser_prompt="benchmark text"\n'
                '[benchmark]\nrounds=5\nwarmup_rounds=2\n', encoding="utf-8")
            completed = subprocess.run([sys.executable, str(Path(bench.__file__)), "--config", str(config_path),
                "--validate", "--protocol", "both", "--mode", "both", "--json", "--output", str(report), "--output-jsonl", str(jsonl),
                "--output-markdown", str(markdown_path)],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["type"], "access_validation")
            self.assertEqual(result, json.loads(report.read_text(encoding="utf-8")))
            self.assertEqual(len(result["probes"]), 2)
            self.assertNotIn("summaries", result)
            self.assertEqual([json.loads(line)["record"] for line in jsonl.read_text(encoding="utf-8").splitlines()], result["probes"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# LLM Access Validation", markdown)
            self.assertIn("## Access Probes", markdown)

    def test_opt_in_endpoint_diagnostics_include_redacted_azure_error_and_request_id(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1",
            auth_type="azure_app_registration", azure_tenant_id="tenant", azure_client_id="app",
            azure_client_secret="private-client-secret")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(bench, "ClientSecretCredential", return_value=credential), \
             patch.object(MockHandler, "variant", "permission_detail"):
            session = bench.RequestSession(config, bench.parse_endpoint(config.endpoint), "test")
            try:
                with self.assertRaises(bench.CallFailure) as failure:
                    session.call(1, streaming=False, diagnostics=True)
            finally:
                session.close()
        details = failure.exception.record["endpoint_error"]
        self.assertEqual(details["code"], "PermissionDenied")
        self.assertIn("Principal lacks inference access", details["message"])
        self.assertEqual(details["request_ids"], {"apim-request-id": "synthetic-request-id"})
        self.assertNotIn("private-client-secret", json.dumps(details))
        self.assertNotIn("fake-azure-access-token", json.dumps(details))

    def test_responses_all_workloads_in_both_response_modes(self):
        for scenario in bench._SCENARIOS:
            variant = "long" if scenario == "long_output" else "normal"
            config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/openai/v1",
                protocol="responses", scenarios=(scenario,), short_requests=2, mode="both")
            requests, paths = [], []
            with self.subTest(scenario=scenario), patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                 patch.object(MockHandler, "variant", variant), patch.object(MockHandler, "requests", requests), \
                 patch.object(MockHandler, "paths", paths):
                result = bench.run_benchmarks(config)
                self.assertTrue(result["success"], [record.get("error") for record in result["runs"]])
                self.assertEqual(set(paths), {"/openai/v1/responses"})
                self.assertTrue(all("input" in request and "messages" not in request for request in requests))
                self.assertEqual(result["comparisons"][0]["protocol"], "responses")
                self.assertEqual(result["comparisons"][0]["paired_rounds"], 1)
                if scenario.startswith("agent"):
                    followups = [request for request in requests if request.get("tool_choice") == "none"]
                    self.assertTrue(followups)
                    for request in followups:
                        self.assertTrue(any(item.get("encrypted_content") == "cipher-test" for item in request["input"]))
                        self.assertTrue(any(item.get("type") == "function_call_output" and item["call_id"] == "call_test" for item in request["input"]))

    def test_both_protocols_alternate_without_mixing_summaries_or_connections(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/openai/v1/",
                             protocol="both", mode="both", rounds=2, warmup_rounds=1)
        saved = []
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config, on_record=saved.append)
        self.assertTrue(result["success"])
        self.assertEqual(result["completed_runs"], 8)
        self.assertEqual(len(result["warmup_runs"]), 4)
        self.assertEqual(saved, result["warmup_runs"] + result["runs"])
        first = [(record["protocol"], record["mode"]) for record in result["runs"][:4]]
        second = [(record["protocol"], record["mode"]) for record in result["runs"][4:]]
        self.assertEqual(first, list(reversed(second)))
        self.assertTrue(all(not record["connection_reused"] for record in result["warmup_runs"]))
        self.assertTrue(all(record["connection_reused"] for record in result["runs"]))
        self.assertEqual(len(result["summaries"]), 4)
        self.assertEqual(len(result["comparisons"]), 2)
        for summary in result["summaries"]:
            self.assertEqual(summary["metrics"]["end_to_end_ms"]["sample_count"], 2)
        for comparison in result["comparisons"]:
            self.assertEqual(comparison["paired_rounds"], 2)

    def test_responses_failures_retain_protocol_and_partial_data(self):
        for scenario in ("baseline", "short_burst", "agent_parallel"):
            for variant in ("http_error", "interrupted", "incomplete_body", "truncated", "malformed_tool"):
                if variant == "malformed_tool" and not scenario.startswith("agent"):
                    continue
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1",
                    protocol="responses", mode="both", scenarios=(scenario,), short_requests=2)
                with self.subTest(scenario=scenario, variant=variant), \
                     patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                     patch.object(MockHandler, "variant", variant):
                    result = bench.run_benchmarks(config)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["comparisons"][0]["paired_rounds"], 0)
                    for record in result["runs"]:
                        self.assertEqual(record["protocol"], "responses")
                        if variant == "interrupted" and record["mode"] == "streaming":
                            self.assertTrue(record["interrupted"])
                            call = record.get("calls", [record])[0]
                            self.assertIn("assistant_message", call)

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_both_http_modes_and_connection_reuse(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions")
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            session = bench.RequestSession(config, bench.parse_endpoint(config.endpoint), "test")
        try:
            streaming = session.call(1)
            non_streaming = session.call(2, streaming=False)
        finally:
            session.close()
        self.assertFalse(streaming["connection_reused"])
        self.assertTrue(non_streaming["connection_reused"])
        self.assertEqual(streaming["tokens"]["output_tokens"], 2)
        self.assertEqual(non_streaming["tokens"]["output_tokens"], 2)
        self.assertNotIn("stream_options", MockHandler.requests[-1])
        self.assertEqual(non_streaming["metrics"]["connection_setup_ms"], 0)
        self.assertEqual(non_streaming["server_timing"]["raw"]["http_server_timing"], "prefill;dur=12")

    def test_incomplete_content_length_never_passes_either_mode(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions", mode="both")
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "incomplete_body"):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(result["failed_runs"], 2)
        self.assertEqual(result["comparisons"][0]["paired_rounds"], 0)
        for record in result["runs"]:
            self.assertIn("IncompleteRead", record["error"])

    def test_interrupted_stream_preserves_data_but_not_success_statistics(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("baseline", "short_burst", "short_conversation", "agent", "agent_serial", "agent_parallel"),
                             short_requests=2)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "interrupted"):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(result["failed_runs"], len(config.scenarios))
        for record in result["runs"]:
            self.assertTrue(record["interrupted"])
            calls = record.get("calls", [record])
            for call in calls:
                self.assertTrue(call["interrupted"])
                self.assertFalse(call["response_complete"])
                self.assertFalse(call["tokens"]["visible_output_tokens_complete"])
                self.assertIsNone(call["metrics"]["ttlt_ms"])
                self.assertIsNone(call["metrics"]["tps"])
                self.assertIsNone(call["metrics"]["end_to_end_ms"])
                self.assertGreater(call["failure_elapsed_ms"], 0)
                self.assertEqual(call["server_timing"]["raw"]["http_server_timing"], "prefill;dur=12")
                if not record["scenario"].startswith("agent"):
                    self.assertEqual(call["assistant_message"]["content"], "hello world")
                    self.assertEqual(call["tokens"]["visible_output_tokens"], 2)
                    self.assertIsNotNone(call["metrics"]["ttft_ms"])
                    self.assertIsNotNone(call["stream"]["last_observed_text_ms"])
            if "calls" in record:
                self.assertFalse(record["visible_token_totals_complete"])
                self.assertIsNone(record["call_metrics"]["ttft_ms"])
        for summary in result["summaries"]:
            self.assertIsNone(summary["metrics"]["ttft_ms"])

    def test_child_failures_retain_completed_parent_and_child_calls(self):
        for scenario in ("agent_serial", "agent_parallel"):
            for variant in ("child_http_error", "child_incomplete_body"):
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                                     scenarios=(scenario,), mode="both")
                with self.subTest(scenario=scenario, variant=variant), \
                     patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                     patch.object(MockHandler, "variant", variant):
                    result = bench.run_benchmarks(config)
                    self.assertEqual(result["failed_runs"], 2)
                    self.assertEqual(result["comparisons"][0]["paired_rounds"], 0)
                    for record in result["runs"]:
                        self.assertGreaterEqual(record["call_count"], 3)
                        self.assertTrue(record["calls"][0]["success"])
                        self.assertFalse(record["visible_token_totals_complete"])
                        failed = [call for call in record["calls"] if not call["success"]]
                        self.assertEqual(len(failed), 1)
                        self.assertEqual(failed[0]["agent"], "subagent:latency")
                        if variant == "child_incomplete_body" and record["mode"] == "streaming":
                            self.assertEqual(failed[0]["assistant_message"]["content"], "hello world")
                            self.assertTrue(failed[0]["interrupted"])

    def test_expired_deadline_does_not_send_request(self):
        with patch.object(bench, "resolve_tokenizer", return_value=None):
            session = bench.RequestSession(make_config(), bench.parse_endpoint(make_config().endpoint), "test")
        with self.assertRaisesRegex(bench.BenchmarkError, "deadline"):
            session.call(1, deadline=-1)

    def test_fresh_connection_policy(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             connection_reuse=False)
        with patch.object(bench, "resolve_tokenizer", return_value=None):
            session = bench.RequestSession(config, bench.parse_endpoint(config.endpoint), "test")
        for number in (1, 2):
            self.assertFalse(session.call(number)["connection_reused"])
            self.assertIsNone(session.connection)

    def test_alternating_comparison_and_warmup_exclusion(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             mode="both", rounds=2, warmup_rounds=1)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["warmup_runs"]), 2)
        self.assertEqual([run["mode"] for run in result["runs"]],
                         ["streaming", "non_streaming", "non_streaming", "streaming"])
        self.assertEqual(result["comparisons"][0]["paired_rounds"], 2)
        self.assertEqual(result["summaries"][0]["metrics"]["ttft_ms"]["sample_count"], 2)

    def test_independent_and_growing_short_batches(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("short_burst", "short_conversation"), short_requests=3)
        MockHandler.requests.clear()
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"])
        self.assertEqual([len(payload["messages"]) for payload in MockHandler.requests], [2, 2, 2, 2, 4, 6])
        self.assertEqual(result["runs"][0]["call_count"], 3)
        self.assertIsNone(result["runs"][0]["metrics"]["ttft_ms"])

    def test_agents_execute_real_tool_round_trips_in_both_modes(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             mode="both", scenarios=("agent", "agent_serial", "agent_parallel"))
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"], [(run.get("error"), run["scenario"]) for run in result["runs"]])
        self.assertEqual([run["call_count"] for run in result["runs"]], [2, 2, 6, 6, 6, 6])
        for run in result["runs"]:
            self.assertEqual(run["tool_support"], "observed")
            self.assertEqual(run["final_answer"], "hello world")
            self.assertLessEqual(run["budget"]["reserved_completion_tokens"], config.agent_total_token_budget)
            self.assertIsNone(run["metrics"]["ttft_ms"])
        delegated = result["runs"][-1]
        self.assertEqual({call["agent"] for call in delegated["calls"]},
                         {"main", "subagent:latency", "subagent:throughput"})

    def test_agent_call_limit_retains_partial_calls(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("agent_serial",), agent_max_calls=2)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(result["runs"][0]["call_count"], 2)
        self.assertIn("maximum LLM call", result["runs"][0]["error"])

    def test_agent_token_budget_is_bounded(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("agent",), agent_total_token_budget=1024)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertIn("token budget exhausted", result["runs"][0]["error"])

    def test_agent_step_limit_is_bounded(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("agent",), agent_max_steps=1)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertIn("maximum step", result["runs"][0]["error"])

    def test_parallel_children_reach_barrier_together(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("agent_parallel",))
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "parallel_barrier", threading.Barrier(2)):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"], result["runs"][0].get("error"))

    def test_long_output_and_oversized_short_via_http(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("long_output",), mode="both")
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "long"):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"])
        self.assertEqual(result["runs"][0]["tokens"]["output_tokens"], 1201)
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "oversized_short"):
            result = bench.run_benchmarks(replace(config, scenarios=("short_burst",), short_requests=2))
        self.assertFalse(result["success"])
        self.assertEqual(result["runs"][0]["call_count"], 2)
        self.assertEqual(result["comparisons"][0]["paired_rounds"], 0)

    def test_endpoint_and_tool_failures_are_reported_without_fallback(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions")
        for variant, scenario in (("http_error", "baseline"), ("interrupted", "baseline"),
                                  ("truncated", "baseline"), ("ignore_tools", "agent"),
                                  ("malformed_tool", "agent")):
            with self.subTest(variant=variant), \
                 patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                 patch.object(MockHandler, "variant", variant):
                result = bench.run_benchmarks(replace(config, scenarios=(scenario,)))
                self.assertFalse(result["success"])
                self.assertTrue(result["runs"][0]["error"])

    def test_cli_json_output_file_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            output_path = Path(directory) / "result.json"
            jsonl_path = Path(directory) / "rounds.jsonl"
            config_path.write_text(f'[llm]\nendpoint = "http://127.0.0.1:{self.server.server_port}/v1/chat/completions"\n'
                'models = ["test"]\nauth_header = ""\n[request]\nuser_prompt = "test"\n'
                '[benchmark]\nrounds = 1\n', encoding="utf-8")
            completed = subprocess.run([sys.executable, str(Path(bench.__file__)), "--config", str(config_path),
                "--mode", "both", "--json", "--output", str(output_path), "--output-jsonl", str(jsonl_path)],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            stdout = json.loads(completed.stdout)
            self.assertEqual(stdout, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertEqual(stdout["completed_runs"], 2)
            lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["record"] for line in lines], stdout["runs"])
            self.assertTrue(all(line["schema_version"] == 2 for line in lines))

    def test_cli_both_protocols_use_base_url_and_persist_distinct_jsonl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            output_path = Path(directory) / "result.json"
            jsonl_path = Path(directory) / "rounds.jsonl"
            markdown_path = Path(directory) / "report.md"
            config_path.write_text(f'[llm]\nendpoint="http://127.0.0.1:{self.server.server_port}/openai/v1"\n'
                'models=["test"]\nauth_header=""\n[request]\nuser_prompt="test"\n[benchmark]\nrounds=1\n', encoding="utf-8")
            completed = subprocess.run([sys.executable, str(Path(bench.__file__)), "--config", str(config_path),
                "--protocol", "both", "--mode", "both", "--json", "--output", str(output_path), "--output-jsonl", str(jsonl_path),
                "--output-markdown", str(markdown_path)],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["completed_runs"], 4)
            self.assertIn("Responses endpoint testing - model=test", completed.stderr)
            self.assertIn("Chat Completions endpoint testing - model=test", completed.stderr)
            self.assertIn("first visible text received", completed.stderr)
            self.assertEqual(result, json.loads(output_path.read_text(encoding="utf-8")))
            lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["record"] for line in lines], result["runs"])
            self.assertEqual({line["record"]["protocol"] for line in lines}, set(bench._PROTOCOL_PATHS))
            for line in lines:
                self.assertTrue(line["endpoint"].endswith(bench._PROTOCOL_PATHS[line["record"]["protocol"]]))
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("## Paired Mode Comparisons", markdown)
            self.assertIn("## Round Results", markdown)
            self.assertIn("## Token Accounting", markdown)

    def test_failed_responses_never_pair_with_successful_chat_completions(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="both", mode="both")
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "responses_http_error"):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(result["failed_runs"], 2)
        self.assertEqual({comparison["protocol"]: comparison["paired_rounds"] for comparison in result["comparisons"]},
                         {"chat_completions": 1, "responses": 0})
        for summary in result["summaries"]:
            self.assertEqual(summary["successful_rounds"], 1 if summary["protocol"] == "chat_completions" else 0)

    def test_responses_conversation_retains_history_but_independent_batches_do_not(self):
        for scenario in ("short_burst", "short_conversation"):
            config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1", protocol="responses",
                                 mode="both", scenarios=(scenario,), short_requests=2)
            requests = []
            with self.subTest(scenario=scenario), patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                 patch.object(MockHandler, "requests", requests):
                result = bench.run_benchmarks(config)
                self.assertTrue(result["success"])
                self.assertEqual([len(request["input"]) for request in requests],
                                 [2, 5, 2, 5] if scenario == "short_conversation" else [2, 2, 2, 2])

    def test_round_callback_includes_warmups_failures_and_actual_execution_order(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             rounds=2, warmup_rounds=1, mode="both")
        records = []
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()):
            result = bench.run_benchmarks(config, on_record=records.append)
        self.assertEqual(records, result["warmup_runs"] + result["runs"])
        self.assertEqual([record["mode"] for record in records],
                         ["streaming", "non_streaming", "streaming", "non_streaming", "non_streaming", "streaming"])
        records.clear()
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "variant", "http_error"):
            result = bench.run_benchmarks(replace(config, continue_on_error=False), on_record=records.append)
        self.assertEqual(records, result["warmup_runs"])
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])

    def test_round_save_failure_stops_requests_even_with_continue_on_error(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             rounds=2, mode="both", continue_on_error=True)
        requests = []
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(MockHandler, "requests", requests), \
             self.assertRaisesRegex(OSError, "disk full"):
            bench.run_benchmarks(config, on_record=Mock(side_effect=OSError("disk full")))
        self.assertEqual(len(requests), 1)

    def test_continue_on_error_false_stops_remaining_modes(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             mode="both", rounds=3, continue_on_error=False)
        with patch.object(bench, "resolve_tokenizer", return_value=None), \
             patch.object(MockHandler, "variant", "http_error"):
            result = bench.run_benchmarks(config)
        self.assertEqual(result["completed_runs"], 1)
        self.assertFalse(result["success"])

    def test_http_failures_are_reported_in_every_scenario(self):
        for scenario in bench._SCENARIOS:
            for continue_on_error in (True, False):
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                    scenarios=(scenario,), mode="both", short_requests=2, continue_on_error=continue_on_error)
                with self.subTest(scenario=scenario, continue_on_error=continue_on_error), \
                     patch.object(bench, "resolve_tokenizer", return_value=None), \
                     patch.object(MockHandler, "variant", "http_error"):
                    result = bench.run_benchmarks(config)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["completed_runs"], 2 if continue_on_error else 1)
                    for record in result["runs"]:
                        self.assertIn("HTTP 400", record["error"])
                        if "calls" in record:
                            self.assertTrue(record["calls"])
                            self.assertFalse(record["visible_token_totals_complete"])
                            self.assertIsNone(record["tokens"]["api_completion_tokens"])

    def test_azure_header_shared_by_modes_and_parallel_subagents(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             scenarios=("agent_parallel",), mode="both", auth_type="azure_managed_identity")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
        captured_headers = []
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(bench, "ManagedIdentityCredential", return_value=credential), \
             patch.object(MockHandler, "auth_headers", captured_headers):
            result = bench.run_benchmarks(config)
        self.assertTrue(result["success"])
        self.assertEqual(captured_headers, ["Bearer fake-azure-access-token"] * 12)
        credential.get_token.assert_called_once()
        credential.close.assert_called_once()
        self.assertNotIn("fake-azure-access-token", json.dumps(result))

    def test_azure_authentication_time_is_outside_model_timing(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             auth_type="azure_managed_identity")
        clock = [0.0]

        def perf_counter():
            clock[0] += 0.01
            return clock[0]

        def get_token(*args, **kwargs):
            clock[0] += 10
            return AccessToken("fake-azure-access-token", 9999999999)

        credential = Mock(spec=["get_token", "close"])
        credential.get_token.side_effect = get_token
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(bench, "ManagedIdentityCredential", return_value=credential), \
             patch.object(bench.time, "perf_counter", side_effect=perf_counter):
            result = bench.run_single_benchmark(config, bench.parse_endpoint(config.endpoint), "test", 1)
        self.assertGreaterEqual(result["metrics"]["authentication_ms"], 10000)
        self.assertLess(result["metrics"]["end_to_end_ms"], 1000)

    def test_auto_fallback_is_reported_in_both_modes_and_parallel_agents(self):
        for app_available in (True, False):
            with self.subTest(app_available=app_available):
                managed = Mock(spec=["get_token", "close"])
                managed.get_token.side_effect = bench.CredentialUnavailableError("not available")
                app = Mock(spec=["get_token", "close"])
                app.get_token.return_value = AccessToken("app-token", 9999999999)
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                    auth_type="auto", api_key="key", auth_header="Authorization", auth_scheme="Bearer",
                    azure_tenant_id="tenant" if app_available else "", azure_client_id="app-client",
                    scenarios=("baseline", "agent_parallel"), mode="both", protocol="both")
                headers = []
                with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                     patch.object(bench, "ManagedIdentityCredential", return_value=managed), \
                     patch.object(bench, "ClientSecretCredential", return_value=app), \
                     patch.dict(os.environ, {"AZURE_CLIENT_SECRET": "private secret"}), \
                     patch.object(MockHandler, "auth_headers", headers):
                    result = bench.run_benchmarks(config)
                self.assertTrue(result["success"])
                self.assertEqual(headers, ["Bearer app-token" if app_available else "Bearer key"] * 28)
                self.assertEqual(result["settings"]["auth_order"], list(bench._AUTH_CHAIN))
                output = io.StringIO()
                with patch.object(sys, "stdout", output):
                    bench.print_human_result(result)
                self.assertIn("Managed Identity -> App Registration -> LLM API key", output.getvalue())
                self.assertIn("auth azure_app_registration" if app_available else "auth api_key", output.getvalue())
                for record in result["runs"]:
                    for call in record.get("calls", [record]):
                        self.assertEqual(call["authentication"]["type"], "azure_app_registration" if app_available else "api_key")
                        self.assertEqual(call["authentication"]["attempts"][0]["type"], "azure_managed_identity")
                managed.get_token.assert_called_once()

    def test_auto_does_not_retry_endpoint_authentication_rejections(self):
        for variant, status in (("unauthorized", 401), ("forbidden", 403)):
            with self.subTest(variant=variant):
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.return_value = AccessToken("managed-token", 9999999999)
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                                     auth_type="auto", api_key="key", auth_header="Authorization")
                with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
                     patch.object(bench, "ManagedIdentityCredential", return_value=credential), \
                     patch.object(bench, "ClientSecretCredential") as app, \
                     patch.object(MockHandler, "variant", variant), \
                     patch.object(MockHandler, "requests", []) as requests:
                    result = bench.run_benchmarks(config)
                self.assertFalse(result["success"])
                self.assertEqual(len(requests), 1)
                self.assertEqual(result["runs"][0]["http_status"], status)
                self.assertEqual(result["runs"][0]["authentication"]["type"], "azure_managed_identity")
                app.assert_not_called()

    def test_auto_exhaustion_reports_attempts_without_sending_requests(self):
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.side_effect = bench.CredentialUnavailableError("fake secret")
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions", auth_type="auto")
        with patch.object(bench, "resolve_tokenizer", return_value=WordTokenizer()), \
             patch.object(bench, "ManagedIdentityCredential", return_value=credential), \
             patch.object(MockHandler, "requests", []) as requests:
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(len(requests), 0)
        self.assertIsNone(result["runs"][0]["authentication"]["type"])
        self.assertEqual(len(result["runs"][0]["authentication"]["attempts"]), 3)
        self.assertNotIn("fake secret", json.dumps(result))

    def test_azure_identity_errors_never_fall_back_or_leak_details(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             auth_type="azure_managed_identity", continue_on_error=False)
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.side_effect = RuntimeError("private-client-secret fake-azure-access-token")
        request_count = len(MockHandler.requests)
        with patch.object(bench, "resolve_tokenizer", return_value=None), \
             patch.object(bench, "ManagedIdentityCredential", return_value=credential):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertEqual(len(MockHandler.requests), request_count)
        self.assertIn("authentication failed", result["runs"][0]["error"])
        self.assertNotIn("private-client-secret", json.dumps(result))
        self.assertNotIn("fake-azure-access-token", json.dumps(result))

    def test_azure_token_and_client_secret_redacted_from_endpoint_error(self):
        config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                             auth_type="azure_app_registration", azure_client_id="app", azure_tenant_id="tenant")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
        with patch.object(bench, "resolve_tokenizer", return_value=None), \
             patch.object(bench, "ClientSecretCredential", return_value=credential), \
             patch.dict(os.environ, {"AZURE_CLIENT_SECRET": "private-client-secret"}), \
             patch.object(MockHandler, "variant", "token_error"):
            result = bench.run_benchmarks(config)
        self.assertFalse(result["success"])
        self.assertNotIn("fake-azure-access-token", json.dumps(result))
        self.assertNotIn("private-client-secret", json.dumps(result))

    def test_inline_client_secret_is_redacted_from_endpoint_error(self):
        for auth_type in ("azure_app_registration", "auto"):
            with self.subTest(auth_type=auth_type):
                config = make_config(endpoint=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                    auth_type=auth_type, azure_client_id="app", azure_tenant_id="tenant",
                    azure_client_secret="private-client-secret", azure_managed_identity_client_id=None)
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
                with patch.object(bench, "resolve_tokenizer", return_value=None), \
                     patch.object(bench, "ClientSecretCredential", return_value=credential), \
                     patch.object(MockHandler, "variant", "token_error"):
                    result = bench.run_benchmarks(config)
                self.assertFalse(result["success"])
                self.assertNotIn("fake-azure-access-token", json.dumps(result))
                self.assertNotIn("private-client-secret", json.dumps(result))


class AuthenticationTests(unittest.TestCase):
    def test_identity_error_preserves_safe_aadsts_codes_and_guidance(self):
        cases = (
            ("AADSTS7000215", "secret value"),
            ("AADSTS7000222", "expired"),
            ("AADSTS700016", "tenant"),
            ("AADSTS90002", "tenant"),
            ("AADSTS500011", "scope"),
            ("AADSTS70011", "scope"),
        )
        for code, guidance in cases:
            with self.subTest(code=code):
                config = make_config(auth_type="azure_app_registration",
                    azure_tenant_id="tenant", azure_client_id="app",
                    azure_client_secret="private-client-secret")
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.side_effect = ClientAuthenticationError(
                    f"{code}: private-client-secret fake-access-token https://identity.example/?secret=private-query")
                auth = bench.Authentication(config)
                with patch.object(bench, "ClientSecretCredential", return_value=credential):
                    try:
                        with self.assertRaises(bench.BenchmarkError) as failure:
                            auth.apply({})
                    finally:
                        auth.close()
                message = str(failure.exception)
                self.assertIn(code, message)
                self.assertIn(guidance, message)
                for sensitive in ("private-client-secret", "fake-access-token", "private-query", "identity.example"):
                    self.assertNotIn(sensitive, message)

    def test_identity_error_redacts_credentials_before_extracting_codes(self):
        for inline in (True, False):
            with self.subTest(inline=inline):
                secret = "AADSTS1234567"
                token = "AADSTS2345678"
                config = make_config(auth_type="azure_app_registration",
                    azure_tenant_id="tenant", azure_client_id="app",
                    azure_client_secret=secret if inline else "")
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.side_effect = ClientAuthenticationError(
                    f"AADSTS7000215: secret={secret} token={token}")
                auth = bench.Authentication(config)
                auth.tokens.add(token)
                with patch.object(bench, "ClientSecretCredential", return_value=credential), \
                     patch.dict(os.environ, {"AZURE_CLIENT_SECRET": secret}):
                    try:
                        with self.assertRaises(bench.BenchmarkError) as failure:
                            auth.apply({})
                    finally:
                        auth.close()
                message = str(failure.exception)
                self.assertIn("AADSTS7000215", message)
                self.assertNotIn(secret, message)
                self.assertNotIn(token, message)

    def test_identity_error_codes_are_bounded_and_deduplicated(self):
        config = make_config(auth_type="azure_app_registration",
            azure_tenant_id="tenant", azure_client_id="app", azure_client_secret="private-secret")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.side_effect = ClientAuthenticationError(
            "AADSTS1234567890 AADSTS1234 xAADSTS54321 AADSTS54321suffix "
            "AADSTS7777777 AADSTS7777777 private-secret")
        auth = bench.Authentication(config)
        with patch.object(bench, "ClientSecretCredential", return_value=credential):
            try:
                with self.assertRaises(bench.BenchmarkError) as failure:
                    auth.apply({})
            finally:
                auth.close()
        message = str(failure.exception)
        self.assertEqual(message.count("AADSTS7777777"), 1)
        for invalid in ("AADSTS1234567890", "AADSTS1234", "AADSTS54321", "private-secret"):
            self.assertNotIn(invalid, message)

    def test_inline_client_secret_is_literal_and_does_not_read_environment(self):
        for auth_type in ("azure_app_registration", "auto"):
            with self.subTest(auth_type=auth_type):
                secret = "fake-secret~${NOT_AN_ENV_REFERENCE}"
                config = make_config(auth_type=auth_type, azure_tenant_id="tenant", azure_client_id="app",
                    azure_client_secret=secret, azure_managed_identity_client_id=None)
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.return_value = AccessToken("fake-access-token", 9999999999)
                auth = bench.Authentication(config)
                environment = Mock()
                environment.get.side_effect = AssertionError("Inline secret must not read environment variables")
                with patch.object(bench.os, "environ", environment), \
                     patch.object(bench, "ClientSecretCredential", return_value=credential) as constructor:
                    headers = {}
                    auth.apply(headers)
                    self.assertEqual(constructor.call_args.kwargs["client_secret"], secret)
                    self.assertEqual(headers, {"Authorization": "Bearer fake-access-token"})
                    self.assertEqual(auth.redact(bench.BenchmarkError(secret)), "[REDACTED]")
                    auth.close()
                self.assertNotIn(secret, repr(config))
                environment.get.assert_not_called()

    def test_auto_uses_managed_identity_first_and_keeps_it_cached(self):
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("managed-token", 9999999999)
        config = make_config(auth_type="auto", api_key="key", auth_header="Authorization",
                             azure_client_id="app-client", azure_managed_identity_client_id="mi-client")
        auth = bench.Authentication(config)
        with patch.object(bench, "ManagedIdentityCredential", return_value=credential) as managed, \
             patch.object(bench, "ClientSecretCredential") as app:
            headers = {}
            auth.apply(headers)
            auth.apply({})
            self.assertEqual(auth.selected_type, "azure_managed_identity")
            self.assertEqual(headers, {"Authorization": "Bearer managed-token"})
            self.assertEqual(managed.call_args.kwargs["client_id"], "mi-client")
            app.assert_not_called()
            credential.get_token.assert_called_once()
            self.assertEqual(len(auth.attempts), 1)
            auth.close()
        credential.close.assert_called_once()

    def test_auto_falls_back_to_app_registration_then_key(self):
        for app_available in (True, False):
            with self.subTest(app_available=app_available):
                managed = Mock(spec=["get_token", "close"])
                managed.get_token.side_effect = bench.CredentialUnavailableError("private identity error")
                app = Mock(spec=["get_token", "close"])
                if app_available:
                    app.get_token.return_value = AccessToken("app-token", 9999999999)
                else:
                    app.get_token.side_effect = ClientAuthenticationError("private secret")
                config = make_config(auth_type="auto", api_key="key", auth_header="api-key", auth_scheme="",
                    azure_tenant_id="tenant", azure_client_id="app-client")
                auth = bench.Authentication(config)
                with patch.object(bench, "ManagedIdentityCredential", return_value=managed), \
                     patch.object(bench, "ClientSecretCredential", return_value=app), \
                     patch.dict(os.environ, {"AZURE_CLIENT_SECRET": "private secret"}):
                    headers = {}
                    auth.apply(headers)
                    self.assertEqual(headers, {"Authorization": "Bearer app-token"} if app_available else {"api-key": "key"})
                    self.assertEqual(auth.selected_type, "azure_app_registration" if app_available else "api_key")
                    self.assertEqual([attempt["type"] for attempt in auth.attempts],
                                     list(bench._AUTH_CHAIN[:2 if app_available else 3]))
                    self.assertNotIn("private", json.dumps(auth.attempts))
                    auth.close()
                managed.close.assert_called_once()
                app.close.assert_called_once()

    def test_auto_skips_missing_app_settings_and_reports_exhaustion(self):
        managed = Mock(spec=["get_token", "close"])
        managed.get_token.side_effect = bench.CredentialUnavailableError("not available")
        auth = bench.Authentication(make_config(auth_type="auto"))
        with patch.object(bench, "ManagedIdentityCredential", return_value=managed), \
             patch.object(bench, "ClientSecretCredential") as app:
            with self.assertRaisesRegex(bench.BenchmarkError, "chain exhausted"):
                auth.apply({})
            with self.assertRaisesRegex(bench.BenchmarkError, "chain exhausted"):
                auth.apply({})
            managed.get_token.assert_called_once()
            app.assert_not_called()
            self.assertEqual(len(auth.attempts), 3)
            auth.close()

    def test_auto_refresh_failure_moves_to_next_available_credential(self):
        managed = Mock(spec=["get_token", "close"])
        managed.get_token.side_effect = [AccessToken("first-token", 10000), bench.CredentialUnavailableError("expired")]
        auth = bench.Authentication(make_config(auth_type="auto", api_key="key", auth_header="Authorization", auth_scheme="Bearer"))
        with patch.object(bench, "ManagedIdentityCredential", return_value=managed):
            with patch.object(bench.time, "time", return_value=1000):
                auth.apply({})
            with patch.object(bench.time, "time", return_value=9800):
                headers = {}
                auth.apply(headers)
            self.assertEqual(headers, {"Authorization": "Bearer key"})
            self.assertEqual(auth.selected_type, "api_key")
            self.assertEqual(auth.redact(bench.BenchmarkError("first-token key")), "[REDACTED] [REDACTED]")
            auth.close()

    def test_actual_sdk_warning_is_sanitized_before_handlers_receive_it(self):
        config = make_config(auth_type="azure_app_registration",
            azure_tenant_id="11111111-1111-1111-1111-111111111111",
            azure_client_id="22222222-2222-2222-2222-222222222222",
            azure_client_secret="fake-client-secret")
        credential = bench.ClientSecretCredential(config.azure_tenant_id, config.azure_client_id,
            "fake-client-secret", logging_enable=False)
        auth = bench.Authentication(config)
        with patch.object(bench, "ClientSecretCredential", return_value=credential), \
             patch.object(credential, "_acquire_token_silently", return_value=None), \
             patch.object(credential, "_request_token", side_effect=ClientAuthenticationError(
                 message="fake-client-secret fake-access-token")), \
             self.assertLogs("azure.identity", level="WARNING") as captured:
            try:
                with self.assertRaisesRegex(bench.BenchmarkError, "client_authentication_error") as failure:
                    auth.apply({})
            finally:
                auth.close()
        for secret in ("fake-client-secret", "fake-access-token"):
            self.assertNotIn(secret, str(failure.exception))
            self.assertNotIn(secret, "\n".join(captured.output))

    def test_identity_log_protection_covers_new_loggers_and_resets_context(self):
        logger = logging.getLogger("azure.identity.review_new_logger")
        with self.assertLogs(logger, level="WARNING") as captured:
            with bench.protected_identity_logging():
                try:
                    raise ValueError("fake-secret-stack")
                except ValueError:
                    logger.exception("fake-secret-message %s", "fake-secret-argument")
            logger.warning("ordinary log outside credential call")
        self.assertNotIn("fake-secret", "\n".join(captured.output))
        self.assertIn("ordinary log outside credential call", captured.output[-1])

    def test_identity_log_context_does_not_hide_other_thread_logs(self):
        logger = logging.getLogger("azure.identity.other_thread")
        with self.assertLogs(logger, level="WARNING") as captured:
            with bench.protected_identity_logging():
                thread = threading.Thread(target=logger.warning, args=("unrelated thread message",))
                thread.start()
                thread.join()
        self.assertIn("unrelated thread message", captured.output[0])

    def test_api_key_authentication_remains_static(self):
        config = make_config(api_key="test-key", auth_header="Authorization", auth_scheme="Bearer")
        _, headers = bench.build_request(config, "test")
        auth = bench.Authentication(config)
        with patch.object(bench, "ClientSecretCredential") as app, \
             patch.object(bench, "ManagedIdentityCredential") as managed:
            self.assertEqual(auth.apply(headers), 0)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        app.assert_not_called()
        managed.assert_not_called()

    def test_app_registration_uses_explicit_client_secret_credential(self):
        config = make_config(auth_type="azure_app_registration", azure_tenant_id="tenant", azure_client_id="app")
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
        auth = bench.Authentication(config)
        with patch.dict(os.environ, {"AZURE_CLIENT_SECRET": "private-client-secret"}), \
             patch.object(bench, "ClientSecretCredential", return_value=credential) as constructor:
            headers = {}
            auth.apply(headers)
            auth.apply({})
            auth.close()
            auth.close()
        self.assertEqual(headers, {"Authorization": "Bearer fake-azure-access-token"})
        self.assertEqual(constructor.call_args.kwargs["tenant_id"], "tenant")
        self.assertEqual(constructor.call_args.kwargs["client_id"], "app")
        self.assertEqual(constructor.call_args.kwargs["client_secret"], "private-client-secret")
        credential.get_token.assert_called_once()
        credential.close.assert_called_once()

    def test_managed_identity_system_and_user_assigned_selection(self):
        for client_id in ("", "user-assigned-client-id"):
            with self.subTest(client_id=client_id):
                credential = Mock(spec=["get_token", "close"])
                credential.get_token.return_value = AccessToken("fake-azure-access-token", 9999999999)
                auth = bench.Authentication(make_config(auth_type="azure_managed_identity", azure_client_id=client_id))
                with patch.object(bench, "ManagedIdentityCredential", return_value=credential) as constructor:
                    auth.apply({})
                    auth.close()
                self.assertEqual(constructor.call_args.kwargs["client_id"], client_id or None)

    def test_sdk_refreshes_token_near_expiry(self):
        credential = Mock(spec=["get_token", "close"])
        credential.get_token.side_effect = [AccessToken("first-fake-token", 10000), AccessToken("second-fake-token", 20000)]
        auth = bench.Authentication(make_config(auth_type="azure_managed_identity"))
        with patch.object(bench, "ManagedIdentityCredential", return_value=credential):
            with patch.object(bench.time, "time", return_value=1000):
                first, cached = {}, {}
                auth.apply(first)
                auth.apply(cached)
            self.assertEqual(credential.get_token.call_count, 1)
            with patch.object(bench.time, "time", return_value=9800):
                refreshed = {}
                auth.apply(refreshed)
            auth.close()
        self.assertEqual(first, cached)
        self.assertEqual(refreshed["Authorization"], "Bearer second-fake-token")
        self.assertEqual(credential.get_token.call_count, 2)


class ScenarioTests(unittest.TestCase):
    def make_record(self, count, source="tiktoken:test"):
        return {"tokens": {"visible_output_tokens": count, "output_token_count_source": source},
                "assistant_message": {"content": "visible text"}, "finish_reason": "stop"}

    def test_strict_length_boundaries(self):
        for scenario, count, passed in (("long_output", 1000, False), ("long_output", 1001, True),
                                         ("short_burst", 199, True), ("short_burst", 200, False)):
            with self.subTest(scenario=scenario, count=count):
                record = self.make_record(count)
                bench.validate_output(record, scenario)
                self.assertEqual(record["success"], passed)

    def test_estimates_do_not_prove_length(self):
        record = self.make_record(1500, "estimated")
        bench.validate_output(record, "long_output")
        self.assertFalse(record["success"])
        self.assertEqual(record["validation"]["status"], "unverified")

    def test_reasoning_budget_setting_is_preserved(self):
        config = make_config(max_tokens=None, max_completion_tokens=256)
        effective = bench.scenario_config(config, "long_output")
        self.assertIsNone(effective.max_tokens)
        self.assertEqual(effective.max_completion_tokens, 8192)

    def test_prefill_context_sizes(self):
        config = make_config(prefill_short_words=16, prefill_long_words=64)
        short = bench.scenario_config(config, "prefill_short").user_prompt
        long = bench.scenario_config(config, "prefill_long").user_prompt
        self.assertEqual(len(long.split()) - len(short.split()), 48)

    def test_error_message_redacts_keys_and_headers(self):
        config = make_config(api_key="private-key", extra_headers={"api-key": "another-key"})
        message = bench.safe_error(ValueError("private-key another-key"), config)
        self.assertEqual(message, "[REDACTED] [REDACTED]")

    def test_aggregate_preserves_successful_call_beside_network_failure(self):
        completed = bench.build_call_result("test", 1, False, bench.Timing(0, completed_at=1),
            bench.StreamResult("hello world", {"completion_tokens": 2}, 0, finish_reason="stop"),
            WordTokenizer(), 200)
        failed = {"success": False, "error": "Connection reset"}
        result = bench.aggregate_calls("test", 1, "non_streaming", [completed, failed], 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["tokens"]["visible_output_tokens"], 2)
        self.assertIsNone(result["tokens"]["api_completion_tokens"])
        self.assertEqual(result["call_count"], 2)
        self.assertFalse(result["visible_token_totals_complete"])


class ProgressTests(unittest.TestCase):
    def test_worker_schedules_five_second_heartbeats_without_delaying_requests(self):
        reporter = bench.ProgressReporter(stream=io.StringIO())
        with patch.object(bench.time, "monotonic", side_effect=[0, 0, 5, 5]), \
             patch.object(reporter.wake, "wait", side_effect=lambda timeout: reporter.stop.set()) as wait, \
             patch.object(reporter, "heartbeat") as heartbeat:
            reporter._run()
        wait.assert_called_once_with(5)
        heartbeat.assert_called_once_with()

    def test_progress_labels_cannot_inject_additional_terminal_lines(self):
        output = io.StringIO()
        reporter = bench.ProgressReporter(stream=output)
        identifier = reporter.begin("model=test\nforged status\r\x1b[31m")
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertNotIn("\x1b", output.getvalue())
        reporter.finish(identifier)

    def test_first_text_callback_is_once_per_stream_and_never_contains_output(self):
        timing = bench.Timing(started_at=0)
        first_text = Mock()
        response = io.BytesIO(b'data: {"choices":[{"delta":{"content":"private answer"},"finish_reason":null}]}\n\n'
                              b'data: {"choices":[{"delta":{"content":" more"},"finish_reason":"stop"}]}\n\n'
                              b'data: [DONE]\n\n')
        result = bench.consume_sse(response, timing, on_first_text=first_text)
        first_text.assert_called_once_with()
        self.assertEqual(result.text, "private answer more")

    def test_stage_updates_are_queued_without_writing_in_measured_reader(self):
        output = io.StringIO()
        reporter = bench.ProgressReporter(stream=output)
        identifier = reporter.begin("stream")
        output.seek(0)
        output.truncate()
        reporter.update(identifier, "first visible text received; generating")
        self.assertEqual(output.getvalue(), "")
        reporter.finish(identifier)
        self.assertIn("first visible text received; generating", output.getvalue())
        self.assertLess(output.getvalue().index("[STATUS"), output.getvalue().index("[DONE"))

    def test_progress_waiting_messages_include_stage_elapsed_time_and_flush(self):
        output = Mock()
        reporter = bench.ProgressReporter(stream=output)
        with patch.object(bench.time, "monotonic", return_value=0) as clock:
            identifier = reporter.begin("Responses endpoint testing - model=test - short output")
            reporter.update(identifier, "waiting for response headers")
            clock.return_value = 10
            reporter.heartbeat()
            reporter.finish(identifier, "PASS")
        text = "".join(call.args[0] for call in output.write.call_args_list)
        self.assertIn("[START #1]", text)
        self.assertIn("waiting for response headers - elapsed 10.0s", text)
        self.assertIn("[PASS #1]", text)
        self.assertGreaterEqual(output.flush.call_count, 3)
        self.assertFalse(reporter.active)

    def test_progress_context_and_thread_are_cleaned_on_interrupt(self):
        output = io.StringIO()
        with bench.ProgressReporter(stream=output) as reporter:
            with self.assertRaises(KeyboardInterrupt), bench.progress_scope("validation"):
                raise KeyboardInterrupt
            self.assertFalse(reporter.active)
        self.assertIsNone(bench._PROGRESS.get())
        self.assertFalse(reporter.thread.is_alive())
        self.assertIn("CANCELLED", output.getvalue())

    def test_disabled_progress_writes_nothing_and_starts_no_thread(self):
        output = io.StringIO()
        with bench.ProgressReporter(enabled=False, stream=output) as reporter:
            with bench.progress_scope("disabled") as update:
                update("PASS")
            reporter.heartbeat()
        self.assertEqual(output.getvalue(), "")
        self.assertIsNone(reporter.thread)


class OutputTests(unittest.TestCase):
    def test_cli_progress_is_on_stderr_and_can_be_disabled(self):
        for options in ((), ("--no-progress",)):
            stderr = io.StringIO()
            with self.subTest(options=options), patch.object(sys, "stderr", stderr), \
                 patch.object(bench, "run_benchmarks", return_value={"success": True}):
                status, result = self.invoke(*options)
            self.assertEqual(status, 0)
            self.assertTrue(result["success"])
            self.assertEqual(bool(stderr.getvalue()), not bool(options))

    def benchmark_result(self):
        config = make_config(rounds=1, protocol="both", mode="both")
        records = []
        for protocol in bench.selected_protocols(config):
            for mode in bench.selected_modes(config):
                record = bench.build_call_result("test", 1, mode == "streaming",
                    bench.Timing(0, request_sent_at=0.5, first_token_at=1 if mode == "streaming" else None,
                        last_token_at=2 if mode == "streaming" else None, completed_at=3),
                    bench.StreamResult("one two", {"completion_tokens": 4}, 3, first_chunk="one ",
                        text_chunk_count=2 if mode == "streaming" else 0, finish_reason="stop"), WordTokenizer(), 200)
                record.update({"scenario": "baseline", "protocol": protocol, "warmup": False})
                bench.validate_output(record, "baseline")
                records.append(record)
        return bench.build_result(config, bench.parse_endpoint(config.endpoint), records)

    def test_markdown_benchmark_tables_preserve_protocols_statistics_and_caveats(self):
        result = self.benchmark_result()
        original = json.dumps(result)
        markdown = bench.render_markdown(result)
        for heading in ("# LLM Benchmark Report", "## Latency Summary (ms)", "## Throughput And Authentication",
                        "## Paired Mode Comparisons", "## Metric Definitions And Limitations"):
            self.assertIn(heading, markdown)
        self.assertIn("chat\\_completions", markdown)
        self.assertIn("responses", markdown)
        self.assertIn("1000.0 / 1000.0 / 1000.0 (n=1)", markdown)
        self.assertIn("n/a means unavailable, not zero", markdown)
        self.assertIn("P95 with fewer than 20", markdown)
        self.assertEqual(json.dumps(result), original)

    def test_markdown_cells_escape_table_breaks_html_and_link_syntax(self):
        table = bench.markdown_table(["Field"], [["a|b\r\n<script>[click](javascript:bad)`*"]])
        self.assertEqual(len(table.splitlines()), 3)
        self.assertEqual(table.splitlines()[-1].count("|"), 2)
        self.assertIn("a&#124;b<br>&lt;script&gt;", table)
        self.assertNotIn("<script>", table)
        self.assertNotIn("[click]", table)

    def test_cli_markdown_output_does_not_change_stdout_json(self):
        result = self.benchmark_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            with patch.object(bench, "run_benchmarks", return_value=result):
                status, stdout = self.invoke("--output-markdown", path)
            self.assertEqual(status, 0)
            self.assertEqual(stdout, result)
            self.assertEqual(path.read_text(encoding="utf-8"), bench.render_markdown(result))

    def test_failed_benchmark_still_saves_markdown_and_returns_failure(self):
        config = make_config(rounds=1)
        failed = {"model": "test", "round": 1, "scenario": "baseline", "mode": "streaming",
                  "success": False, "error": "Endpoint returned HTTP 401 PermissionDenied"}
        result = bench.build_result(config, bench.parse_endpoint(config.endpoint), [failed])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.md"
            with patch.object(bench, "run_benchmarks", return_value=result):
                status, stdout = self.invoke("--output-markdown", path)
            self.assertEqual(status, 1)
            self.assertFalse(stdout["success"])
            markdown = path.read_text(encoding="utf-8")
            self.assertIn("PARTIAL FAILURE", markdown)
            self.assertIn("HTTP 401 PermissionDenied", markdown)
            self.assertIn("## Failures And Length Validation", markdown)

    def test_invalid_markdown_output_fails_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "report.md"
            with patch.object(bench, "run_benchmarks") as runner:
                status, result = self.invoke("--output-markdown", path)
            self.assertEqual(status, 1)
            self.assertFalse(result["success"])
            runner.assert_not_called()

    def test_markdown_retains_failed_warmups_partial_tokens_and_agent_calls(self):
        config = make_config(rounds=1, scenarios=("agent",))
        failed_call = {"model": "test", "round": 1, "mode": "streaming", "scenario": "agent",
                      "success": False, "error": "HTTP 401 | denied\n<script>"}
        task = bench.aggregate_calls("test", 1, "streaming", [failed_call], 0)
        task.update({"scenario": "agent", "protocol": "chat_completions", "warmup": False})
        warmup = {**failed_call, "warmup": True}
        result = bench.build_result(config, bench.parse_endpoint(config.endpoint), [task], [warmup])
        markdown = bench.render_markdown(result)
        for heading in ("## Individual LLM Calls", "## Task Details", "## Token Accounting", "## Failures And Length Validation"):
            self.assertIn(heading, markdown)
        self.assertIn("warmup", markdown)
        self.assertIn("HTTP 401 &#124; denied<br>&lt;script&gt;", markdown)
        self.assertIn("observed portions", markdown)
        self.assertIn("n/a", markdown)
        self.assertNotIn("<script>", markdown)
        self.assertIn("| Failed warmups | 1 |", markdown)

    def test_markdown_report_is_atomic_on_interrupt_and_disk_failure(self):
        result = self.benchmark_result()
        for mode in ("preflight", "final_sync", "interrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.md"
                path.write_text("previous report", encoding="utf-8")
                sync = [OSError("disk full")] if mode == "preflight" else [None, OSError("disk full")] if mode == "final_sync" else [None]
                with patch.object(bench, "run_benchmarks", return_value=result,
                                  side_effect=KeyboardInterrupt if mode == "interrupt" else None) as runner, \
                     patch.object(bench.os, "fsync", side_effect=sync):
                    status, _ = self.invoke("--output-markdown", path)
                self.assertEqual(status, 130 if mode == "interrupt" else 1)
                self.assertEqual(runner.call_count, 0 if mode == "preflight" else 1)
                self.assertEqual(path.read_text(encoding="utf-8"), "previous report")
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_markdown_cannot_alias_config_json_or_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text("existing config", encoding="utf-8")
            alias = Path(directory) / "alias.md"
            alias.symlink_to(config_path)
            for arguments in (("--config", config_path, "--output-markdown", alias),
                              ("--output", alias, "--output-markdown", alias),
                              ("--output-jsonl", alias, "--output-markdown", alias)):
                with self.subTest(arguments=arguments), patch.object(bench, "run_benchmarks") as runner:
                    status, _ = self.invoke(*arguments)
                self.assertEqual(status, 1)
                runner.assert_not_called()
            self.assertEqual(config_path.read_text(encoding="utf-8"), "existing config")

    def invoke(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, "argv", ["llm_connection_test.py", "--json", *map(str, arguments)]), \
             patch.object(bench, "load_config", return_value=make_config()), patch.object(sys, "stdout", output):
            status = bench.main()
        return status, json.loads(output.getvalue())

    def test_invalid_output_paths_fail_before_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text("original config", encoding="utf-8")
            alias = Path(directory) / "alias.json"
            alias.symlink_to(config_path)
            hardlink = Path(directory) / "hardlink.json"
            hardlink.hardlink_to(config_path)
            for path in (Path(directory) / "missing" / "result.json", Path(directory), config_path, alias, hardlink):
                with self.subTest(path=path), patch.object(bench, "run_benchmarks") as runner:
                    status, result = self.invoke("--config", config_path, "--output", path)
                    self.assertEqual(status, 1)
                    self.assertFalse(result["success"])
                    runner.assert_not_called()
            self.assertEqual(config_path.read_text(encoding="utf-8"), "original config")

    def test_json_output_is_replaced_only_after_benchmark_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("previous report", encoding="utf-8")

            def run(config):
                self.assertEqual(path.read_text(encoding="utf-8"), "previous report")
                return {"success": True, "completed_runs": 2}

            with patch.object(bench, "run_benchmarks", side_effect=run):
                status, result = self.invoke("--output", path)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_preflight_or_final_sync_preserves_previous_report(self):
        for failures, calls in (([OSError("disk full")], 0), ([None, OSError("disk full")], 1)):
            with self.subTest(calls=calls), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.json"
                path.write_text("previous report", encoding="utf-8")
                with patch.object(bench, "run_benchmarks", return_value={"success": True}) as runner, \
                     patch.object(bench.os, "fsync", side_effect=failures):
                    status, result = self.invoke("--output", path)
                self.assertEqual(status, 1)
                self.assertIn("disk full", result["error"])
                self.assertEqual(runner.call_count, calls)
                self.assertEqual(path.read_text(encoding="utf-8"), "previous report")
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_keyboard_interrupt_preserves_previous_report_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("previous report", encoding="utf-8")
            with patch.object(bench, "run_benchmarks", side_effect=KeyboardInterrupt):
                status, result = self.invoke("--output", path)
            self.assertEqual(status, 130)
            self.assertIn("interrupted", result["error"])
            self.assertEqual(path.read_text(encoding="utf-8"), "previous report")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_jsonl_is_readable_before_next_round_and_survives_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            report = Path(directory) / "report.json"
            report.write_text("previous report", encoding="utf-8")
            record = {"model": "test", "round": 1, "warmup": False, "success": False, "error": "HTTP 400"}

            def run(config, *, on_record):
                on_record(record)
                lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([line["record"] for line in lines], [record])
                self.assertNotIn("?", lines[0]["endpoint"])
                raise KeyboardInterrupt

            with patch.object(bench, "run_benchmarks", side_effect=run):
                status, result = self.invoke("--output", report, "--output-jsonl", path)
            self.assertEqual(status, 130)
            self.assertIn("retained", result["error"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["record"], record)
            self.assertEqual(report.read_text(encoding="utf-8"), "previous report")
            self.assertEqual(set(Path(directory).iterdir()), {path, report})

    def test_jsonl_paths_are_preflighted_and_existing_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            path.write_text("previous data", encoding="utf-8")
            for target in (path, Path(directory), Path(directory) / "missing" / "rounds.jsonl"):
                with self.subTest(target=target), patch.object(bench, "run_benchmarks") as runner:
                    status, result = self.invoke("--output-jsonl", target)
                    self.assertEqual(status, 1)
                    self.assertFalse(result["success"])
                    runner.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "previous data")
            with patch.object(bench, "run_benchmarks") as runner:
                status, _ = self.invoke("--output", path, "--output-jsonl", path)
            self.assertEqual(status, 1)
            runner.assert_not_called()

    def test_jsonl_sync_failure_stops_execution_and_preserves_completed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            record = {"round": 1, "success": True}
            syncs = [0]

            def sync(descriptor):
                syncs[0] += 1
                if syncs[0] == 4:
                    raise OSError("disk full")

            def run(config, *, on_record):
                on_record(record)
                on_record({"round": 2, "success": True})
                self.fail("must stop when the sink fails")

            with patch.object(bench, "run_benchmarks", side_effect=run), patch.object(bench.os, "fsync", side_effect=sync):
                status, failure = self.invoke("--output-jsonl", path)
            self.assertEqual(status, 1)
            self.assertIn("disk full", failure["error"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8").splitlines()[0])["record"], record)

    def test_failed_jsonl_preflight_sync_removes_probe_and_sends_no_request(self):
        for failures in ([OSError("disk full")], [None, OSError("disk full")]):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rounds.jsonl"
                with patch.object(bench, "run_benchmarks") as runner, \
                     patch.object(bench.os, "fsync", side_effect=failures):
                    status, result = self.invoke("--output-jsonl", path)
                self.assertEqual(status, 1)
                self.assertIn("disk full", result["error"])
                runner.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_check_config_does_not_create_output_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bench, "run_benchmarks") as runner:
            status, result = self.invoke("--check-config", "--output", Path(directory) / "report.json",
                                         "--output-jsonl", Path(directory) / "rounds.jsonl",
                                         "--output-markdown", Path(directory) / "report.md")
            self.assertEqual(status, 0)
            self.assertTrue(result["success"])
            runner.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


class ConfigurationTests(unittest.TestCase):
    def test_protocol_config_and_offline_endpoint_resolution(self):
        for protocol in bench._PROTOCOLS:
            with self.subTest(protocol=protocol):
                config = self.load_document(llm=f'protocol="{protocol}"', endpoint="https://example.com/openai/v1/",
                                            require_credentials=False)
                self.assertEqual(config.protocol, protocol)
        for value in ('"invalid"', "true", "1"):
            with self.subTest(value=value), self.assertRaises(bench.ConfigurationError):
                self.load_document(llm="protocol=" + value, require_credentials=False)

    def load_document(self, benchmark="", request="", llm="", require_credentials=True,
                      azure_auth="", endpoint="http://localhost/v1/chat/completions"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(f'[llm]\nendpoint="{endpoint}"\nmodels=["test"]\n'
                f'auth_header=""\n{llm}\n[request]\nuser_prompt="test"\n{request}\n[benchmark]\n{benchmark}\n'
                f'[azure_auth]\n{azure_auth}\n', encoding="utf-8")
            return bench.load_config(path, require_credentials=require_credentials)

    def test_invalid_config_values_are_rejected(self):
        for settings in ('rounds=true', 'warmup_rounds=-1', 'mode="invalid"',
                         'scenarios=["missing"]', 'scenarios=["baseline","baseline"]',
                         'task_timeout_seconds=nan'):
            with self.subTest(settings=settings), self.assertRaises(bench.ConfigurationError):
                self.load_document(benchmark=settings)
        with self.assertRaises(bench.ConfigurationError):
            self.load_document(request="max_tokens=1\nmax_completion_tokens=1")

    def test_invalid_ports_and_malformed_hosts_fail_offline_validation(self):
        for endpoint in ("http://localhost:99999/v1/chat/completions", "http://localhost:no-port/v1/chat/completions",
                         "http://localhost:0/v1/chat/completions", "http://localhost:/v1/chat/completions",
                         "http://[invalid/v1/chat/completions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(bench.ConfigurationError):
                self.load_document(endpoint=endpoint, require_credentials=False)

    def test_valid_ipv6_and_explicit_ports_are_accepted(self):
        self.assertEqual(bench.parse_endpoint("http://[::1]:8080/v1/chat/completions").port, 8080)
        self.assertEqual(bench.parse_endpoint("https://localhost:65535/v1/chat/completions").port, 65535)

    def test_cli_reports_invalid_port_without_connecting(self):
        document = (b'[llm]\nendpoint="http://localhost:99999/v1/chat/completions"\nmodels=["test"]\n'
                    b'auth_header=""\n[request]\nuser_prompt="test"\n')
        for options in (["--json"], ["--check-config", "--json"]):
            output = io.StringIO()
            with self.subTest(options=options), patch.object(Path, "open", return_value=io.BytesIO(document)), \
                 patch.object(sys, "argv", ["llm_connection_test.py", *options]), \
                 patch.object(sys, "stdout", output), patch.object(bench, "run_benchmarks") as runner:
                self.assertEqual(bench.main(), 1)
                runner.assert_not_called()
            failure = json.loads(output.getvalue())
            self.assertFalse(failure["success"])
            self.assertIn("port", failure["error"])

    def test_check_config_does_not_resolve_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self.load_document(llm='api_key_env="UNSET_BENCHMARK_TEST_KEY"', require_credentials=False)
        self.assertEqual(config.api_key, "")

    def test_unknown_explicit_tokenizer_is_not_silently_estimated(self):
        with self.assertRaises(bench.ConfigurationError):
            bench.resolve_tokenizer(make_config(tokenizer_model="unknown-tokenizer-test"), "test")

    def test_azure_config_check_does_not_read_credentials(self):
        for auth_type in ("azure_app_registration", "azure_managed_identity", "auto"):
            with self.subTest(auth_type=auth_type), patch.dict(os.environ, {}, clear=True):
                config = self.load_document(llm=f'auth_type="{auth_type}"\napi_key_env="UNSET_LLM_KEY"',
                    endpoint="https://example.com/v1/chat/completions", require_credentials=False)
            self.assertEqual(config.auth_type, auth_type)
            self.assertEqual(config.api_key, "")

    def test_auto_missing_credentials_do_not_preempt_available_sources(self):
        for azure_settings in ("", 'tenant_id=""\nclient_id=""\nclient_secret_env=""'):
            with self.subTest(azure_settings=azure_settings), patch.dict(os.environ, {}, clear=True):
                config = self.load_document(llm='auth_type="auto"\napi_key_env="MISSING_KEY"',
                    azure_auth=azure_settings, endpoint="https://example.com/v1/chat/completions")
            self.assertEqual(config.api_key, "")
            self.assertEqual(config.azure_tenant_id, "")
            self.assertEqual(config.azure_client_id, "")
            self.assertEqual(config.azure_managed_identity_client_id, "")

    def test_auto_resolves_app_and_managed_identity_ids_independently(self):
        with patch.dict(os.environ, {"AZURE_TENANT_ID": "tenant", "AZURE_CLIENT_ID": "app-client",
                                    "MI_CLIENT_ID": "mi-client", "LLM_KEY": "fake-key"}, clear=True):
            config = self.load_document(llm='auth_type="auto"\napi_key_env="LLM_KEY"',
                azure_auth='managed_identity_client_id="${MI_CLIENT_ID}"',
                endpoint="https://example.com/v1/chat/completions")
        self.assertEqual(config.api_key, "fake-key")
        self.assertEqual(config.azure_tenant_id, "tenant")
        self.assertEqual(config.azure_client_id, "app-client")
        self.assertEqual(config.azure_managed_identity_client_id, "mi-client")

    def test_missing_user_assigned_id_does_not_use_system_identity_in_auto(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self.load_document(llm='auth_type="auto"',
                azure_auth='managed_identity_client_id="${MISSING_MI_CLIENT_ID}"',
                endpoint="https://example.com/v1/chat/completions")
        self.assertIsNone(config.azure_managed_identity_client_id)
        with patch.object(bench, "ManagedIdentityCredential") as managed, \
             self.assertRaisesRegex(bench.BenchmarkError, "chain exhausted"):
            bench.Authentication(config).apply({})
        managed.assert_not_called()

    def test_auto_rejects_malformed_auth_configuration(self):
        for llm, azure in (('api_key_env="invalid-name"', ""),
                           ('api_key_env="KEY"\napi_key="another"', ""),
                           ("", 'client_secret_env="invalid-name"')):
            with self.subTest(llm=llm, azure=azure), self.assertRaises(bench.ConfigurationError):
                self.load_document(llm='auth_type="auto"\n' + llm, azure_auth=azure,
                    endpoint="https://example.com/v1/chat/completions", require_credentials=False)

    def test_app_registration_reads_ids_but_does_not_store_secret(self):
        with patch.dict(os.environ, {"AZURE_TENANT_ID": "tenant", "AZURE_CLIENT_ID": "app",
                                     "AZURE_CLIENT_SECRET": "private-client-secret"}, clear=True):
            config = self.load_document(llm='auth_type="azure_app_registration"\napi_key_env="UNSET_LLM_KEY"',
                endpoint="https://example.com/v1/chat/completions")
        self.assertEqual(config.azure_tenant_id, "tenant")
        self.assertEqual(config.azure_client_id, "app")
        self.assertNotIn("private-client-secret", repr(config))

    def test_inline_client_secret_loads_without_environment_and_is_hidden_from_repr(self):
        for auth_type in ("azure_app_registration", "auto"):
            with self.subTest(auth_type=auth_type), patch.dict(os.environ, {}, clear=True):
                config = self.load_document(llm=f'auth_type="{auth_type}"',
                    azure_auth='tenant_id="tenant"\nclient_id="app"\nclient_secret="fake-secret~${LITERAL}"',
                    endpoint="https://example.com/v1/chat/completions")
            self.assertEqual(config.azure_client_secret, "fake-secret~${LITERAL}")
            self.assertEqual(config.azure_client_secret_env, "")
            self.assertNotIn("fake-secret", repr(config))

    def test_inline_client_secret_and_environment_source_are_mutually_exclusive(self):
        with self.assertRaisesRegex(bench.ConfigurationError, "not both"):
            self.load_document(llm='auth_type="auto"', require_credentials=False,
                azure_auth='client_secret="fake-secret"\nclient_secret_env="AZURE_CLIENT_SECRET"',
                endpoint="https://example.com/v1/chat/completions")

    def test_managed_identity_needs_no_secret_or_tenant(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self.load_document(llm='auth_type="azure_managed_identity"',
                endpoint="https://example.com/v1/chat/completions")
        self.assertEqual(config.azure_client_id, "")
        self.assertEqual(config.azure_tenant_id, "")

    def test_azure_rejects_insecure_or_invalid_config(self):
        for settings in ('scope="invalid"', 'client_secret=123'):
            with self.subTest(settings=settings), self.assertRaises(bench.ConfigurationError):
                self.load_document(llm='auth_type="azure_managed_identity"', azure_auth=settings,
                    endpoint="https://example.com/v1/chat/completions", require_credentials=False)
        with self.assertRaisesRegex(bench.ConfigurationError, "HTTPS"):
            self.load_document(llm='auth_type="azure_managed_identity"', require_credentials=False)
        with self.assertRaises(bench.ConfigurationError):
            self.load_document(llm='auth_type="invalid"')


if __name__ == "__main__":
    unittest.main()