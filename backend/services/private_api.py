"""Layer 4 — Private info handler: akses data pribadi mahasiswa via API UNHAS.

Aktif hanya saat UNHAS_API_BASE_URL dikonfigurasi. Jika tidak, fall back ke RAG.
Menggunakan LLM function calling untuk ekstrak parameter dari query natural language.
"""

import logging
from typing import Optional

import httpx

from backend.config import UNHAS_API_BASE_URL
from backend.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = [
    {
        "name": "get_ipk",
        "description": "Ambil IPK mahasiswa yang sedang login",
        "params": [],
    },
    {
        "name": "get_nilai",
        "description": "Ambil nilai mata kuliah tertentu milik mahasiswa",
        "params": ["mata_kuliah", "semester"],
    },
    {
        "name": "get_jadwal",
        "description": "Ambil jadwal kuliah mahasiswa",
        "params": ["semester"],
    },
    {
        "name": "get_krs",
        "description": "Ambil Kartu Rencana Studi mahasiswa semester ini",
        "params": [],
    },
    {
        "name": "get_tagihan",
        "description": "Ambil informasi tagihan UKT mahasiswa",
        "params": [],
    },
]

_TOOL_SELECTION_PROMPT = """Kamu adalah sistem yang mengidentifikasi endpoint API yang tepat.

Endpoint tersedia:
{tools}

Query pengguna: "{query}"

Pilih endpoint yang paling sesuai dan ekstrak parameter yang dibutuhkan.
Format respons (JSON saja, tidak ada teks lain):
{{"tool": "<nama_tool>", "params": {{"<key>": "<value>"}}}}

Jika tidak ada endpoint yang cocok, balas: {{"tool": null}}"""


def is_available() -> bool:
    return bool(UNHAS_API_BASE_URL)


def handle_private_query(question: str, user_token: str) -> Optional[dict]:
    """
    Proses query get_info_private.
    Returns dict result atau None jika API tidak tersedia (caller lanjut ke RAG).
    """
    if not UNHAS_API_BASE_URL:
        logger.info("private_api: not configured, falling back to RAG")
        return None

    tool_call = _extract_tool_call(question)
    if not tool_call or not tool_call.get("tool"):
        logger.info("private_api: no matching tool for query")
        return {
            "answer": "Maaf, saya tidak dapat mengidentifikasi data pribadi yang diminta. "
                      "Silakan hubungi bagian akademik secara langsung.",
            "sources": [],
            "mode": "get_info_private_unmatched",
        }

    tool_name = tool_call["tool"]
    params = tool_call.get("params", {})
    endpoint = f"/api/me/{tool_name.replace('get_', '')}"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{UNHAS_API_BASE_URL}{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {user_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return {
                "answer": "Sesi Anda telah berakhir. Silakan login ulang.",
                "sources": [],
                "mode": "get_info_private_auth_error",
            }
        logger.error(f"private_api_http_error status={e.response.status_code}")
        return {
            "answer": "Gagal mengambil data dari sistem UNHAS. Coba beberapa saat lagi.",
            "sources": [],
            "mode": "get_info_private_error",
        }
    except Exception as e:
        logger.error(f"private_api_error error={e}")
        return None  # fall back to RAG

    answer = _format_private_data(tool_name, data)
    return {
        "answer": answer,
        "sources": [],
        "mode": "get_info_private",
    }


def _extract_tool_call(question: str) -> Optional[dict]:
    """LLM memilih tool dan mengekstrak parameter dari query."""
    import json
    tools_str = "\n".join(
        f"- {t['name']}: {t['description']}"
        + (f" (params: {', '.join(t['params'])})" if t["params"] else "")
        for t in AVAILABLE_TOOLS
    )
    prompt = _TOOL_SELECTION_PROMPT.format(tools=tools_str, query=question)
    llm = get_llm(temperature=0.0)
    raw = str(llm.complete(prompt)).strip()

    # Extract JSON from response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


def _format_private_data(tool_name: str, data: dict) -> str:
    """Format data API menjadi jawaban natural language."""
    if tool_name == "get_ipk":
        ipk = data.get("ipk", "N/A")
        sks = data.get("total_sks", "N/A")
        return f"IPK Anda saat ini adalah **{ipk}** dengan total **{sks} SKS** yang telah ditempuh."
    if tool_name == "get_krs":
        courses = data.get("courses", [])
        if not courses:
            return "KRS Anda untuk semester ini belum diisi."
        lines = [f"- {c.get('name', 'N/A')} ({c.get('sks', '?')} SKS)" for c in courses]
        return "KRS semester ini:\n" + "\n".join(lines)
    # Generic fallback
    return f"Data berhasil diambil: {data}"
