"""
Tests for the root application endpoint.
Covers: health-check response, trace-ID propagation.
"""


class TestRootEndpoint:
    """Tests for GET /."""

    def test_read_main(self, client):
        """Root endpoint should return a running status message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "MRLN Arcane Tuner API is running" in response.json()["message"]
        assert "X-Trace-ID" in response.headers

    def test_trace_id_propagation(self, client):
        """Client-supplied X-Trace-ID should be echoed back."""
        trace_id = "test-trace-id-123"
        response = client.get("/", headers={"X-Trace-ID": trace_id})
        assert response.status_code == 200
        assert response.headers["X-Trace-ID"] == trace_id
