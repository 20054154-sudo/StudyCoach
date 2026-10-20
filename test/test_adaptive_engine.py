"""Unit test — backend/agents/adaptive_engine.py (logica pura, nessun I/O)."""

from backend.agents.adaptive_engine import AdaptiveEngine, TopicStats, _build_summary, SessionStats


# ---------------------------------------------------------------------------
# TopicStats
# ---------------------------------------------------------------------------

def test_topic_stats_avg_score_empty():
    ts = TopicStats(topic="ml")
    assert ts.avg_score == 0.0
    assert ts.n_attempts == 0
    assert ts.is_weak is False


def test_topic_stats_is_weak_single_low_score():
    ts = TopicStats(topic="ml", scores=[3])
    assert ts.is_weak is True


def test_topic_stats_is_weak_average_below_six():
    ts = TopicStats(topic="ml", scores=[7, 4])
    assert ts.avg_score == 5.5
    assert ts.is_weak is True


def test_topic_stats_not_weak_average_at_or_above_six():
    ts = TopicStats(topic="ml", scores=[6, 6])
    assert ts.is_weak is False


def test_topic_stats_single_score_below_five_is_weak():
    ts = TopicStats(topic="ml", scores=[4])
    assert ts.is_weak is True


def test_topic_stats_single_score_at_or_above_six_not_weak():
    ts = TopicStats(topic="ml", scores=[6])
    assert ts.is_weak is False


# ---------------------------------------------------------------------------
# AdaptiveEngine.record_answer
# ---------------------------------------------------------------------------

def test_record_answer_creates_topic_and_clamps_score():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="reti neurali", score=15)
    stats = engine.stats.topic_stats["reti neurali"]
    assert stats.scores == [10]


def test_record_answer_clamps_negative_score():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="x", score=-5)
    assert engine.stats.topic_stats["x"].scores == [0]


def test_record_answer_accumulates_multiple_scores_same_topic():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="x", score=8)
    engine.record_answer(question_id=2, topic="x", score=4)
    assert engine.stats.topic_stats["x"].scores == [8, 4]
    assert engine.stats.answered_question_ids == [1, 2]


# ---------------------------------------------------------------------------
# AdaptiveEngine.pick_next_question
# ---------------------------------------------------------------------------

def test_pick_next_question_returns_none_if_no_remaining():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    assert engine.pick_next_question([]) is None


def test_pick_next_question_prioritizes_weak_topic():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="debole", score=2)
    engine.record_answer(question_id=2, topic="forte", score=9)

    remaining = [
        {"id": 10, "topic_hint": "forte"},
        {"id": 11, "topic_hint": "debole"},
    ]
    chosen = engine.pick_next_question(remaining)
    assert chosen["id"] == 11


def test_pick_next_question_prioritizes_unseen_topic_over_seen_strong():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="visto", score=9)

    remaining = [
        {"id": 10, "topic_hint": "visto"},
        {"id": 11, "topic_hint": "nuovo"},
    ]
    chosen = engine.pick_next_question(remaining)
    assert chosen["id"] == 11


def test_pick_next_question_falls_back_to_first_remaining():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="visto", score=9)

    remaining = [{"id": 10, "topic_hint": "visto"}]
    chosen = engine.pick_next_question(remaining)
    assert chosen["id"] == 10


# ---------------------------------------------------------------------------
# AdaptiveEngine.generate_session_report
# ---------------------------------------------------------------------------

def test_generate_session_report_structure_and_sorting():
    engine = AdaptiveEngine(student_id=42, exam_id=7)
    engine.record_answer(question_id=1, topic="forte", score=9)
    engine.record_answer(question_id=2, topic="debole", score=3)

    report = engine.generate_session_report()

    assert report["student_id"] == 42
    assert report["exam_id"] == 7
    assert report["n_questions_answered"] == 2
    assert report["weak_topics"] == ["debole"]
    assert report["strong_topics"] == ["forte"]
    # Ordinato per avg_score crescente: debole (3) prima di forte (9)
    assert [t["topic"] for t in report["topic_breakdown"]] == ["debole", "forte"]
    assert report["overall_avg"] == 6.0


def test_generate_session_report_empty_session():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    report = engine.generate_session_report()
    assert report["overall_avg"] == 0.0
    assert report["n_questions_answered"] == 0
    assert report["weak_topics"] == []
    assert "Nessuna domanda completata" in report["summary"]


def test_sufficient_topic_is_not_reported_as_mastered():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="tema", score=6)
    report = engine.generate_session_report()
    assert report["weak_topics"] == []
    assert report["strong_topics"] == []


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

def test_build_summary_high_average():
    stats = SessionStats(student_id=1, exam_id=1, topic_stats={"x": TopicStats(topic="x", scores=[9, 8])})
    summary = _build_summary(stats)
    assert "Ottima prestazione" in summary
    assert "padroneggiati" in summary


def test_build_summary_low_average_mentions_weak_topics():
    stats = SessionStats(student_id=1, exam_id=1, topic_stats={"x": TopicStats(topic="x", scores=[2, 3])})
    summary = _build_summary(stats)
    assert "insufficiente" in summary
    assert "rivedere" in summary
    assert "'x'" in summary


def test_to_json_serializes_stats():
    engine = AdaptiveEngine(student_id=1, exam_id=1)
    engine.record_answer(question_id=1, topic="x", score=7)
    payload = engine.to_json()
    assert '"student_id": 1' in payload
    assert '"x"' in payload
