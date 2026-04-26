# Drill 6B — Embedding Basics

Load, query, and compare pre-trained GloVe word embeddings.

## Objectives

- Load pre-trained GloVe vectors from a text file into a Python dictionary
- Compute cosine similarity between word vectors
- Find nearest neighbors in embedding space

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Tasks

Complete the three functions in `drill.py`:

1. **`load_glove(filepath)`** — Load the provided GloVe file into a dictionary mapping words to numpy arrays
2. **`cosine_similarity(vec1, vec2)`** — Compute cosine similarity between two vectors
3. **`nearest_neighbors(word, embeddings, n=5)`** — Find the top-n most similar words to a query word

## Submission

1. Create a branch named `drill-6b-embedding-basics`
2. Complete all functions in `drill.py`
3. Run `pytest tests/ -v` to verify your work
4. Open a PR to `main` — the autograder will run automatically
5. Paste your PR URL into TalentLMS → Module 6 Week B → Drill 6B to submit this assignment

Resubmissions are accepted through Saturday of the assignment week.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
