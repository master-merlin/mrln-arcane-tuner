import httpx
from app.core.llm.openai_compat import chat_vision


class _CaptureTransport(httpx.BaseTransport):
    def __init__(self):
        self.payload = None

    def handle_request(self, request):
        import json

        self.payload = json.loads(request.content)
        body = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        return httpx.Response(200, json=body)


def test_chat_vision_forwards_response_format():
    t = _CaptureTransport()
    out = chat_vision(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        prompt="p",
        image_jpeg=b"\xff\xd8\xff",
        response_format={"type": "json_object"},
        transport=t,
    )
    assert out == '{"ok":true}'
    assert t.payload["response_format"] == {"type": "json_object"}


def test_chat_vision_omits_response_format_when_none():
    t = _CaptureTransport()
    chat_vision(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        prompt="p",
        image_jpeg=b"\xff\xd8\xff",
        transport=t,
    )
    assert "response_format" not in t.payload
