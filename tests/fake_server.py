"""没有 API Key 时预览界面用：用假模型启动后端（回复是固定的，只用来检查界面）。

    python -X utf8 -m uvicorn tests.fake_server:app --port 8766
"""
from counselor import server
from counselor.config import load_config
from counselor.llm import FakeLLM
from counselor.session import Counselor, Indexes
from tests.test_pipeline import fake_handler

server._counselor = Counselor(FakeLLM(fake_handler()), load_config(), Indexes())
app = server.app
