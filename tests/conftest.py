import sys
import types
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "phenopatient"
sys.path.insert(0, str(SOURCE_DIR))


gateway = types.ModuleType("JD_gateway_caller")
gateway.analyze_text = lambda *args, **kwargs: ""
sys.modules.setdefault("JD_gateway_caller", gateway)
