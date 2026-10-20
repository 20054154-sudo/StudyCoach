
import json
from dataclasses import dataclass, field, asdict
from statistics import mean


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------

@dataclass
class TopicStats:
    """Statistiche di un singolo topic per uno studente in una sessione."""
    topic: str
    scores: list[int] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def n_attempts(self) -> int:
        return len(self.scores)

    @property
    def is_weak(self) -> bool:
        """Topic debole se media < 6 o un solo tentativo con score < 5."""
        if not self.scores:
            return False
        if len(self.scores) == 1 and self.scores[0] < 5:
            return True
        return self.avg_score < 6.0


@dataclass
class SessionStats:
    """Statistiche aggregate di una sessione completa."""
    student_id: int
    exam_id: int
    topic_stats: dict[str, TopicStats] = field(default_factory=dict)
    answered_question_ids: list[int] = field(default_factory=list)

    @property
    def overall_avg(self) -> float:
        all_scores = [s for t in self.topic_stats.values() for s in t.scores]
        return mean(all_scores) if all_scores else 0.0

    @property
    def weak_topics(self) -> list[str]:
        return [t for t, s in self.topic_stats.items() if s.is_weak]

    @property
    def strong_topics(self) -> list[str]:
        return [t for t, s in self.topic_stats.items() if s.avg_score >= 8]


# ---------------------------------------------------------------------------
# Adaptive Engine
# ---------------------------------------------------------------------------

class AdaptiveEngine:


    def __init__(self, student_id: int, exam_id: int):
        self.stats = SessionStats(student_id=student_id, exam_id=exam_id)

    def record_answer(self, question_id: int, topic: str, score: int) -> None:

        score = max(0, min(10, score))

        if topic not in self.stats.topic_stats:
            self.stats.topic_stats[topic] = TopicStats(topic=topic)

        self.stats.topic_stats[topic].scores.append(score)
        self.stats.answered_question_ids.append(question_id)

        print(
            f"[AdaptiveEngine] Registrato: q_id={question_id} | "
            f"topic='{topic}' | score={score}/10 | "
            f"avg_topic={self.stats.topic_stats[topic].avg_score:.1f}"
        )

    def pick_next_question(self, remaining: list[dict]) -> dict | None:

        if not remaining:
            return None

        weak = set(self.stats.weak_topics)

        # Priorità 1: domande su topic deboli
        weak_questions = [q for q in remaining if q.get("topic_hint") in weak]
        if weak_questions:
            chosen = weak_questions[0]
            print(
                f"[AdaptiveEngine] Prossima domanda su topic DEBOLE: "
                f"'{chosen.get('topic_hint')}' | q_id={chosen.get('id')}"
            )
            return chosen

        # Priorità 2: domande su topic non ancora visti
        seen_topics = set(self.stats.topic_stats.keys())
        unseen = [q for q in remaining if q.get("topic_hint") not in seen_topics]
        if unseen:
            chosen = unseen[0]
            print(
                f"[AdaptiveEngine] Prossima domanda su topic NUOVO: "
                f"'{chosen.get('topic_hint')}' | q_id={chosen.get('id')}"
            )
            return chosen

        # Priorità 3: qualsiasi domanda rimanente
        return remaining[0]

    def generate_session_report(self) -> dict:

        stats = self.stats

        topic_breakdown = []
        for topic, ts in sorted(
            stats.topic_stats.items(),
            key=lambda x: x[1].avg_score
        ):
            topic_breakdown.append({
                "topic": topic,
                "avg_score": round(ts.avg_score, 1),
                "n_attempts": ts.n_attempts,
                "is_weak": ts.is_weak,
            })

        report = {
            "student_id": stats.student_id,
            "exam_id": stats.exam_id,
            "overall_avg": round(stats.overall_avg, 1),
            "n_questions_answered": len(stats.answered_question_ids),
            "weak_topics": stats.weak_topics,
            "strong_topics": stats.strong_topics,
            "topic_breakdown": topic_breakdown,
            "summary": _build_summary(stats),
        }

        print(
            f"[AdaptiveEngine] Report generato | "
            f"media={report['overall_avg']} | "
            f"topic deboli={report['weak_topics']}"
        )
        return report

    def to_json(self) -> str:
        """Serializza le statistiche correnti (per salvataggio intermedio)."""
        d = asdict(self.stats)
        return json.dumps(d, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Funzione interna per il testo del summary
# ---------------------------------------------------------------------------

def _build_summary(stats: SessionStats) -> str:
    """Costruisce un testo di riepilogo leggibile per il report finale."""
    if not stats.topic_stats:
        return "Nessuna domanda completata in questa sessione."

    lines = []
    avg = stats.overall_avg

    if avg >= 8:
        lines.append("Ottima prestazione complessiva.")
    elif avg >= 6:
        lines.append("Prestazione sufficiente con margini di miglioramento.")
    else:
        lines.append("Prestazione insufficiente: è necessario rivedere il materiale.")

    if stats.weak_topics:
        topics_str = ", ".join(f"'{t}'" for t in stats.weak_topics)
        lines.append(f"Argomenti da rivedere: {topics_str}.")

    if stats.strong_topics:
        topics_str = ", ".join(f"'{t}'" for t in stats.strong_topics)
        lines.append(f"Argomenti padroneggiati: {topics_str}.")

    return " ".join(lines)
