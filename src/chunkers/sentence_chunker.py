from pandas import DataFrame


from src.chunkers.base import BaseChunker
from src.utils.nlp import split_into_sentences


class SentenceChunker(BaseChunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.nlp = spacy.blank("en")  # Or load a specific spaCy model

    def chunk(self, texts_df: DataFrame, **kwargs) -> DataFrame:
        """
        Splits texts in a DataFrame into individual sentences and enumerates them.

        Input DataFrame columns (must be present):
            - "report_id": identifier for each document/text unit
            - "text": full text for the corresponding report/document,
                      which may consist of multiple sentences

        Returns:
            DataFrame with columns:
                - "report_id": identifier inherited from input
                - "sentence_id": integer, starts at 1 for each report_id
                - "sentence": text content of the sentence

            The original "text" column is dropped.

        Example:
            Input:
                report_id | text
                1         | "This is a test. Second sentence."
            Output:
                report_id | sentence_id | sentence
                1         | 1           | "This is a test."
                1         | 2           | "Second sentence."
        """
        records = []
        for _, row in texts_df.iterrows():
            report_id = row["report_id"]
            sentences = split_into_sentences(row["text"])
            for sent_id, sent in enumerate(sentences):
                records.append(
                    {
                        "report_id": report_id,
                        "sentence_id": sent_id + 1,
                        "sentence": str(sent),
                    }
                )
        return DataFrame(records)
