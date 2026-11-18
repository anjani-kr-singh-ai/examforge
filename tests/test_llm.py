import pytest

from examforge import prompts
from examforge.generation import base_vars
from examforge.llm import LLMClient, LLMError, OutputError, TransientLLMError
from examforge.schemas import GenerationRequest, JudgeScores
from tests.conftest import ScriptedProvider

OK = JudgeScores(relevance=5, factual_correctness=5, clarity=5, difficulty_fit=5, distractor_quality=5, answer_explanation_consistency=5, rationale="ok")


def client(script, retries=3):
    sleeps = []
    c = LLMClient(ScriptedProvider(script), max_retries=retries, sleep=sleeps.append)
    return c, sleeps


def run(c):
    req = GenerationRequest(chapter="Input Tax Credit", topic="Blocked credits")
    v = base_vars(req) | {"question_block": "Q", "context": "ctx"}
    return c.run("judge", JudgeScores, v)


def test_transient_failures_retry_with_backoff():
    c, sleeps = client({"JudgeScores": [TransientLLMError("429", retry_after=7), TransientLLMError("503"), OK]})
    assert run(c) == OK
    assert c.calls[-1].attempts == 3 and c.calls[-1].ok and sleeps[0] == 7 and len(sleeps) == 2


def test_malformed_output_gets_structured_correction():
    c, _ = client({"JudgeScores": [OutputError("output failed schema validation: clarity: field required"), OK]})
    assert run(c) == OK
    second_prompt = c.provider.seen[1][1]
    assert "CORRECTION REQUIRED" in second_prompt and "clarity: field required" in second_prompt


def test_retries_are_bounded_and_logged():
    c, _ = client({"JudgeScores": [OutputError("truncated")] * 5}, retries=2)
    with pytest.raises(LLMError, match="failed after 2 attempts"):
        run(c)
    assert not c.calls[-1].ok and c.calls[-1].attempts == 2 and c.calls[-1].errors


def test_fatal_errors_are_not_retried():
    c, sleeps = client({"JudgeScores": [LLMError("API error 401"), OK]})
    with pytest.raises(LLMError, match="401"):
        run(c)
    assert c.calls[-1].attempts == 1 and not sleeps


def test_every_prompt_version_renders_with_pipeline_variables():
    req = GenerationRequest(chapter="Input Tax Credit", topic="Blocked credits")
    v = base_vars(req) | {k: "x" for k in ("context", "avoid", "items", "question_json", "issues", "question_block",
                                            "question_blind", "evidence_block")}
    stages = [p.name for p in prompts.PROMPT_DIR.iterdir() if p.is_dir()]
    assert len(stages) == 7
    for stage in stages:
        for ver in prompts.versions(stage):
            system, user = prompts.render(stage, ver, v)
            assert "$" not in system.replace("$$", "") and system and user


def test_missing_prompt_variable_fails_loudly():
    with pytest.raises(KeyError):
        prompts.render("judge", "v1", {"subject": "IDT"})


def test_bedrock_provider_parses_wrapped_json_and_classifies_errors():
    from botocore.exceptions import ClientError

    from examforge.llm import BedrockProvider

    class Stub:
        def __init__(self, out):
            self.out = out

        def converse(self, **kw):
            if isinstance(self.out, Exception):
                raise self.out
            return self.out

    def reply(text, stop="end_turn"):
        return {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": text}]}},
                "stopReason": stop, "usage": {"inputTokens": 10, "outputTokens": 5}}

    ok = BedrockProvider("m", client=Stub(reply("<think>x</think>```json\n" + OK.model_dump_json() + "\n```")))
    assert ok.generate("s", "u", JudgeScores, {}).parsed == OK
    with pytest.raises(OutputError):
        BedrockProvider("m", client=Stub(reply("{", "max_tokens"))).generate("s", "u", JudgeScores, {})
    with pytest.raises(OutputError):
        BedrockProvider("m", client=Stub(reply('{"relevance": 5}'))).generate("s", "u", JudgeScores, {})
    throttle = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")
    with pytest.raises(TransientLLMError):
        BedrockProvider("m", client=Stub(throttle)).generate("s", "u", JudgeScores, {})
    denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse")
    with pytest.raises(LLMError) as e:
        BedrockProvider("m", client=Stub(denied)).generate("s", "u", JudgeScores, {})
    assert not isinstance(e.value, TransientLLMError)
