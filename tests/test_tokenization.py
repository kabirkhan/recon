from recon.tokenization import add_tokens


def test_add_tokens():

    bad_example = {
        "text": "The primary outcome was spontaneous delivery before 34 weeks.face=+Italic; Resultsface=-Italic; Spontaneous delivery before 34 weeks of gestation was less frequent in the progesterone group than in the placebo group (19.2% vs. 34.4%; relative risk, 0.56; 95% confidence interval [CI], 0.36 to 0.86).",
        "spans": [
            {"text": "34 weeks", "start": 61, "end": 69, "label": "DURATION"},
            {"text": "gestation", "start": 133, "end": 145, "label": "DURATION"},
            {"text": "19.2%", "start": 226, "end": 231, "label": "PERCENT"},
            {"text": "34.4%", "start": 236, "end": 241, "label": "PERCENT"},
            {"text": "0.56", "start": 258, "end": 262, "label": "NUMBER"},
            {"text": "95%", "start": 264, "end": 267, "label": "PERCENT"},
            {"text": "0.36 to 0.86", "start": 294, "end": 306, "label": "NUM_RANGE"},
        ],
    }

    fixed_example = add_tokens([bad_example])[0]

    assert fixed_example["tokens"]
    assert fixed_example["spans"][0]["token_start"] == 7
    assert fixed_example["spans"][0]["token_end"] == 8

    assert fixed_example["spans"][1]["token_start"] == 17
    assert fixed_example["spans"][1]["token_end"] == 17
