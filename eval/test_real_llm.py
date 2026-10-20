import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import run


@pytest.mark.real_llm
def test_real_llm_eval_produces_dated_results():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY non configurata")
    output = run(validate_only=False)
    assert output.exists()
    assert output.parent.name == "results"
