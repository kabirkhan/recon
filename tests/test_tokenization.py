from recon.operations import op_iter
from recon.preprocess import SpacyPreProcessor
from recon.tokenization import add_tokens
from recon.types import Example


def test_add_tokens(spacy_preprocessor):
    # fmt: off
    untokenized_examples = [Example(**example) for example in [
        {"text": "Have you used the new version of my model?", "spans": [{"start": 36, "end": 41, "label": "SKILL"}]},
        {"text": "I'd like to work as an actor or model if possible.", "spans": [{"text": "actor", "start": 23, "end": 28, "label": "JOB_ROLE"}, {"text": "model", "start": 32, "end": 37, "label": "JOB_ROLE"}]},
        {"text": "We are looking for a Software Development Engineer who has solid coding skills, a strong machine learning background, and is passionate about developing new AI products.", "spans": [{"start": 21, "end": 50, "label": "SKILL"}, {"start": 65, "end": 71, "label": "SKILL"}, {"start": 89, "end": 105, "label": "SKILL"}, {"start": 142, "end": 152, "label": "SKILL"}, {"start": 157, "end": 159, "label": "SKILL"}]},
        {"text": "Responsibilities As a SOFTWARE DEVELOPMENT ENGINEER II you will work / collaborate with other talented engineers to build features and technologies that will affect millions of your fellow developers in the community.", "spans": [{"start": 22, "end": 51, "label": "JOB_ROLE"}, {"start": 71, "end": 82, "label": "SKILL"}, {"start": 103, "end": 112, "label": "JOB_ROLE"}, {"start": 135, "end": 147, "label": "SKILL"}, {"start": 189, "end": 199, "label": "JOB_ROLE"}]}
    ]]

    tokenized_examples = [Example(**example) for example in [
        {"text": "Have you used the new version of my model?", "spans": [{"start": 36, "end": 41, "token_start": 8, "token_end": 8, "label": "SKILL"}], "tokens": [{"text": "Have", "start": 0, "end": 4, "id": 0}, {"text": "you", "start": 5, "end": 8, "id": 1}, {"text": "used", "start": 9, "end": 13, "id": 2}, {"text": "the", "start": 14, "end": 17, "id": 3}, {"text": "new", "start": 18, "end": 21, "id": 4}, {"text": "version", "start": 22, "end": 29, "id": 5}, {"text": "of", "start": 30, "end": 32, "id": 6}, {"text": "my", "start": 33, "end": 35, "id": 7}, {"text": "model", "start": 36, "end": 41, "id": 8}, {"text": "?", "start": 41, "end": 42, "id": 9}]},
        {"text": "I'd like to work as an actor or model if possible.", "spans": [{"text": "actor", "start": 23, "end": 28, "token_start": 7, "token_end": 7, "label": "JOB_ROLE"}, {"text": "model", "start": 32, "end": 37, "token_start": 9, "token_end": 9, "label": "JOB_ROLE"}], "tokens": [{"text": "I", "start": 0, "end": 1, "id": 0}, {"text": "'d", "start": 1, "end": 3, "id": 1}, {"text": "like", "start": 4, "end": 8, "id": 2}, {"text": "to", "start": 9, "end": 11, "id": 3}, {"text": "work", "start": 12, "end": 16, "id": 4}, {"text": "as", "start": 17, "end": 19, "id": 5}, {"text": "an", "start": 20, "end": 22, "id": 6}, {"text": "actor", "start": 23, "end": 28, "id": 7}, {"text": "or", "start": 29, "end": 31, "id": 8}, {"text": "model", "start": 32, "end": 37, "id": 9}, {"text": "if", "start": 38, "end": 40, "id": 10}, {"text": "possible", "start": 41, "end": 49, "id": 11}, {"text": ".", "start": 49, "end": 50, "id": 12}]},
        {"text": "We are looking for a Software Development Engineer who has solid coding skills, a strong machine learning background, and is passionate about developing new AI products.", "tokens": [{"text": "We", "start": 0, "end": 2, "id": 0}, {"text": "are", "start": 3, "end": 6, "id": 1}, {"text": "looking", "start": 7, "end": 14, "id": 2}, {"text": "for", "start": 15, "end": 18, "id": 3}, {"text": "a", "start": 19, "end": 20, "id": 4}, {"text": "Software", "start": 21, "end": 29, "id": 5}, {"text": "Development", "start": 30, "end": 41, "id": 6}, {"text": "Engineer", "start": 42, "end": 50, "id": 7}, {"text": "who", "start": 51, "end": 54, "id": 8}, {"text": "has", "start": 55, "end": 58, "id": 9}, {"text": "solid", "start": 59, "end": 64, "id": 10}, {"text": "coding", "start": 65, "end": 71, "id": 11}, {"text": "skills", "start": 72, "end": 78, "id": 12}, {"text": ",", "start": 78, "end": 79, "id": 13}, {"text": "a", "start": 80, "end": 81, "id": 14}, {"text": "strong", "start": 82, "end": 88, "id": 15}, {"text": "machine", "start": 89, "end": 96, "id": 16}, {"text": "learning", "start": 97, "end": 105, "id": 17}, {"text": "background", "start": 106, "end": 116, "id": 18}, {"text": ",", "start": 116, "end": 117, "id": 19}, {"text": "and", "start": 118, "end": 121, "id": 20}, {"text": "is", "start": 122, "end": 124, "id": 21}, {"text": "passionate", "start": 125, "end": 135, "id": 22}, {"text": "about", "start": 136, "end": 141, "id": 23}, {"text": "developing", "start": 142, "end": 152, "id": 24}, {"text": "new", "start": 153, "end": 156, "id": 25}, {"text": "AI", "start": 157, "end": 159, "id": 26}, {"text": "products", "start": 160, "end": 168, "id": 27}, {"text": ".", "start": 168, "end": 169, "id": 28}], "spans": [{"start": 21, "end": 50, "token_start": 5, "token_end": 7, "label": "SKILL"}, {"start": 65, "end": 71, "token_start": 11, "token_end": 11, "label": "SKILL"}, {"start": 89, "end": 105, "token_start": 16, "token_end": 17, "label": "SKILL"}, {"start": 142, "end": 152, "token_start": 24, "token_end": 24, "label": "SKILL"}, {"start": 157, "end": 159, "token_start": 26, "token_end": 26, "label": "SKILL"}]},
        {"text": "Responsibilities As a SOFTWARE DEVELOPMENT ENGINEER II you will work / collaborate with other talented engineers to build features and technologies that will affect millions of your fellow developers in the community.", "tokens": [{"text": "Responsibilities", "start": 0, "end": 16, "id": 0}, {"text": "As", "start": 17, "end": 19, "id": 1}, {"text": "a", "start": 20, "end": 21, "id": 2}, {"text": "SOFTWARE", "start": 22, "end": 30, "id": 3}, {"text": "DEVELOPMENT", "start": 31, "end": 42, "id": 4}, {"text": "ENGINEER", "start": 43, "end": 51, "id": 5}, {"text": "II", "start": 52, "end": 54, "id": 6}, {"text": "you", "start": 55, "end": 58, "id": 7}, {"text": "will", "start": 59, "end": 63, "id": 8}, {"text": "work", "start": 64, "end": 68, "id": 9}, {"text": "/", "start": 69, "end": 70, "id": 10}, {"text": "collaborate", "start": 71, "end": 82, "id": 11}, {"text": "with", "start": 83, "end": 87, "id": 12}, {"text": "other", "start": 88, "end": 93, "id": 13}, {"text": "talented", "start": 94, "end": 102, "id": 14}, {"text": "engineers", "start": 103, "end": 112, "id": 15}, {"text": "to", "start": 113, "end": 115, "id": 16}, {"text": "build", "start": 116, "end": 121, "id": 17}, {"text": "features", "start": 122, "end": 130, "id": 18}, {"text": "and", "start": 131, "end": 134, "id": 19}, {"text": "technologies", "start": 135, "end": 147, "id": 20}, {"text": "that", "start": 148, "end": 152, "id": 21}, {"text": "will", "start": 153, "end": 157, "id": 22}, {"text": "affect", "start": 158, "end": 164, "id": 23}, {"text": "millions", "start": 165, "end": 173, "id": 24}, {"text": "of", "start": 174, "end": 176, "id": 25}, {"text": "your", "start": 177, "end": 181, "id": 26}, {"text": "fellow", "start": 182, "end": 188, "id": 27}, {"text": "developers", "start": 189, "end": 199, "id": 28}, {"text": "in", "start": 200, "end": 202, "id": 29}, {"text": "the", "start": 203, "end": 206, "id": 30}, {"text": "community", "start": 207, "end": 216, "id": 31}, {"text": ".", "start": 216, "end": 217, "id": 32}], "spans": [{"start": 22, "end": 51, "token_start": 3, "token_end": 5, "label": "JOB_ROLE"}, {"start": 71, "end": 82, "token_start": 11, "token_end": 11, "label": "SKILL"}, {"start": 103, "end": 112, "token_start": 15, "token_end": 15, "label": "JOB_ROLE"}, {"start": 135, "end": 147, "token_start": 20, "token_end": 20, "label": "SKILL"}, {"start": 189, "end": 199, "token_start": 28, "token_end": 28, "label": "JOB_ROLE"}]}
    ]]
    # fmt: on

    fixed_examples = []
    for orig_example_hash, example, preprocessed_outputs in op_iter(
        untokenized_examples, pre=[spacy_preprocessor]
    ):
        fixed_examples.append(add_tokens(example, preprocessed_outputs=preprocessed_outputs))

    for fixed_example, tokenized_example in zip(fixed_examples, tokenized_examples):
        assert fixed_example.text == tokenized_example.text
        assert fixed_example.spans == tokenized_example.spans
        assert fixed_example.tokens == tokenized_example.tokens


def test_add_tokens_bad_example(spacy_preprocessor):

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

    examples = [Example(text=bad_example["text"], spans=[]), Example(**bad_example)]

    fixed_examples = []
    for orig_example_hash, example, preprocessed_outputs in op_iter(
        examples, pre=[spacy_preprocessor]
    ):
        fixed_example = add_tokens(example, preprocessed_outputs=preprocessed_outputs)
        if fixed_example:
            fixed_examples.append(fixed_example)

    assert len(fixed_examples) == 1
    fixed_example = fixed_examples[0]

    # Since add_tokens cannot resolve token start and ends from the spans above.
    assert isinstance(fixed_example, Example)

    assert isinstance(fixed_example.tokens, list)
    assert len(fixed_example.tokens) == 56
