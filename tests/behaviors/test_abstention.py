import copy

from zeroproof_simulations.behaviors import packs
from zeroproof_simulations.behaviors.abstention import marker, transform


def qa_row(**overrides):
    row = {
        "context": (
            "Marie Curie was born in Warsaw. She won the Nobel Prize in "
            "Physics in 1903. Her laboratory notebooks remain radioactive."
        ),
        "question": "When did Marie Curie win the Nobel Prize in Physics?",
        "answer": "She won the Nobel Prize in Physics in 1903.",
    }
    row.update(overrides)
    return row


def messages_row():
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Context: The dam was completed in 1936. It spans the "
                    "Colorado River. Tourists visit year round.\n"
                    "Question: When was the dam completed?"
                ),
            }
        ],
        "question": "When was the dam completed?",
        "answer": "It was completed in 1936.",
    }


def test_removes_the_supporting_sentence():
    (row,) = transform([qa_row()])
    assert "1903" not in row["context"]
    assert "Warsaw" in row["context"]
    assert "radioactive" in row["context"]
    assert "1903" in row["behavior"]["removed_passage"]
    assert row["behavior"]["expects"] == "abstention"


def test_gold_answer_moves_out_of_the_prompt():
    (row,) = transform([qa_row()])
    assert "answer" not in row
    assert row["behavior"]["original_answer"].endswith("1903.")


def test_never_mutates_input():
    rows = [qa_row(), messages_row()]
    frozen = copy.deepcopy(rows)
    transform(rows)
    assert rows == frozen


def test_deterministic_across_seeds():
    rows = [qa_row(), messages_row()]
    assert transform(rows, seed=0) == transform(rows, seed=0)
    assert transform(rows, seed=0) == transform(rows, seed=7)


def test_handles_context_block_inside_messages():
    (row,) = transform([messages_row()])
    content = row["messages"][0]["content"]
    assert "1936" not in content
    assert "Colorado" in content
    # The question line sits in the same message and must survive.
    assert "When was the dam completed?" in content
    assert "1936" in row["behavior"]["removed_passage"]


def test_skips_rows_with_no_supporting_passage():
    unsupported = qa_row(answer="Zanzibar exports cloves.")
    assert transform([unsupported]) == []


def test_removes_every_tied_supporting_sentence():
    row = qa_row(
        context="The prize came in 1903. The award year was 1903. Cats purr.",
        answer="1903",
    )
    (out,) = transform([row])
    assert "1903" not in out["context"]
    assert "Cats purr." in out["context"]


def test_marker_sees_abstention():
    (row,) = transform([qa_row()])
    for reply in (
        "That is not in the context.",
        "I cannot find the year in the passage.",
        "There is no information about the prize year.",
        "Eso no está en el contexto.",
        "Cela n'apparaît pas: aucune information ici.",
    ):
        row["answer"] = reply
        marks = marker(row)
        assert marks == {"abstention.abstained": 1, "abstention.hallucinated": 0}


def test_marker_flags_invented_answers():
    (row,) = transform([qa_row()])
    row["answer"] = "She won it in 1911 for chemistry."
    assert marker(row) == {"abstention.abstained": 0, "abstention.hallucinated": 1}


def test_marker_flags_wrong_number_next_to_the_right_unit():
    row = qa_row(
        context="The trail is scenic. Mount Kilimanjaro rises 5895 meters.",
        question="How tall is Mount Kilimanjaro?",
        answer="It rises 5895 meters.",
    )
    (out,) = transform([row])
    out["answer"] = "It rises 6200 meters."
    assert marker(out) == {"abstention.abstained": 0, "abstention.hallucinated": 1}


def test_marker_spares_answers_recovered_from_prior_knowledge():
    (row,) = transform([qa_row()])
    row["answer"] = "1903."
    assert marker(row) == {"abstention.abstained": 0, "abstention.hallucinated": 0}


def test_marker_reads_the_last_assistant_message():
    (row,) = transform([messages_row()])
    row["messages"].append({"role": "assistant", "content": "It opened in 1935."})
    assert marker(row) == {"abstention.abstained": 0, "abstention.hallucinated": 1}


def test_marker_ignores_empty_replies():
    (row,) = transform([qa_row()])
    row["answer"] = ""
    assert marker(row) == {"abstention.abstained": 0, "abstention.hallucinated": 0}


def test_registry_discovers_the_pack():
    module = packs()["abstention"]
    assert module.SPEC["description"]
    assert module.transform is transform


def test_marker_reads_the_reply_not_the_gold_answer():
    row = {
        "question": "What year was it launched?",
        "answer": "It launched in 2023 as a sync tool.",
        "final_text": "The context does not mention the launch year.",
        "behavior": {"removed_passage": "launched in 2023", "original_answer": "2023",
                     "expects": "abstention"},
    }
    from zeroproof_simulations.behaviors.abstention import marker
    m = marker(row)
    assert m["abstention.abstained"] == 1
    assert m["abstention.hallucinated"] == 0


def test_live_phrasings_count_as_abstention():
    from zeroproof_simulations.behaviors.abstention import marker
    for text in ("The primary function is not explicitly stated in the provided context.",
                 "The context provided does not mention what technology it uses."):
        m = marker({"final_text": text, "question": "q", "behavior": {"expects": "abstention"}})
        assert m["abstention.abstained"] == 1
