import pandas as pd
import pytest

from src.chunkers.sentence_chunker import SentenceChunker


class DummyEmbedder:
    def __call__(self, texts):
        # Dummy embedding returns an array of zeros with appropriate shape
        import torch

        return torch.zeros((len(texts), 3))


@pytest.fixture
def test_input_df():
    return pd.DataFrame(
        [
            {
                "report_id": 1,
                "text": (
                    "Cystic formations with characteristics similar to the "
                    "one described, although smaller in size, were "
                    "observed both in a location adjacent to the main one "
                    "and randomly distributed within the adrenal "
                    "parenchyma. With these findings, the diagnosis of "
                    "cystic lymphangioma was made."
                ),
            },
            {
                "report_id": 2,
                "text": ("This injury was confirmed by CT and scintigraphy."),
            },
            {
                "report_id": 3,
                "text": (
                    "Surgical intervention was decided, during which a "
                    "mass 2 cm in size, well defined, with a surprising red "
                    "color, was found, which was easily dissected from the "
                    "upper testicular and epididymal pole, preserving the "
                    "testis. Histologically, splenic tissue was diagnosed "
                    "without microscopic alterations, with preserved "
                    "architecture and presence of white pulp with germinal "
                    "centers, red pulp with venous sinuses and Billroth "
                    "cords. At 16 months the patient is asymptomatic with a "
                    "physical examination within normal limits."
                ),
            },
        ]
    )


def test_sentence_chunking(test_input_df):
    chunker = SentenceChunker(
        strategy="sentence",
        embedding_pt_path="dummy.pt",
        index_csv_path="dummy.csv",
        llm_embedder=DummyEmbedder(),
        split="test",
    )
    result_df = chunker.chunk(test_input_df)
    expected_rows = [
        {
            "report_id": 1,
            "sentence_id": 1,
            "sentence": (
                "Cystic formations with characteristics similar to the one "
                "described, although smaller in size, were observed both in "
                "a location adjacent to the main one and randomly "
                "distributed within the adrenal parenchyma."
            ),
        },
        {
            "report_id": 1,
            "sentence_id": 2,
            "sentence": (
                "With these findings, the diagnosis of cystic lymphangioma "
                "was made."
            ),
        },
        {
            "report_id": 2,
            "sentence_id": 1,
            "sentence": "This injury was confirmed by CT and scintigraphy.",
        },
        {
            "report_id": 3,
            "sentence_id": 1,
            "sentence": (
                "Surgical intervention was decided, during which a "
                "mass 2 cm in size, well defined, with a surprising red "
                "color, was found, which was easily dissected from the "
                "upper testicular and epididymal pole, preserving the "
                "testis."
            ),
        },
        {
            "report_id": 3,
            "sentence_id": 2,
            "sentence": (
                "Histologically, splenic tissue was diagnosed without "
                "microscopic alterations, with preserved architecture and "
                "presence of white pulp with germinal centers, red pulp with "
                "venous sinuses and Billroth cords."
            ),
        },
        {
            "report_id": 3,
            "sentence_id": 3,
            "sentence": (
                "At 16 months the patient is asymptomatic with a physical "
                "examination within normal limits."
            ),
        },
    ]
    expected_df = pd.DataFrame(expected_rows)

    # Reset index for comparison
    result_df = result_df.reset_index(drop=True)
    print("result_df = ")
    print(result_df)
    expected_df = expected_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(result_df, expected_df)
