from fastapi.testclient import TestClient

from api.main import app, get_pipeline


class _FakePipeline:
    def ask(self, question, k=None):
        return {
            "question": question,
            "answer": "NIST recommends VPNs for remote access [1].",
            "refused": False,
            "citations": [1],
            "invalid_citations": [],
            "input_flags": [],
            "sources": [{"n": 1, "source": "nist_sp800_82r3.pdf", "page": 140}],
        }


app.dependency_overrides[get_pipeline] = lambda: _FakePipeline()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_grounded_answer():
    r = client.post("/ask", json={"question": "remote access to OT?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == [1]
    assert body["sources"][0]["source"].startswith("nist")
    assert body["refused"] is False


def test_ask_rejects_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length validation
