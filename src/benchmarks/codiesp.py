import numpy as np
import os
import pandas as pd
import simple_icd_10_cm as icd
from typing import List
import sys


sys.path.append("..")

from src.benchmarks.base import Benchmark, BenchmarkFactory
from src.benchmarks.data import BenchmarkData

BENCHMARK_FILE_CTRS = {"train": 500, "test": 250}


@BenchmarkFactory.register("CodiEsp")
class CodiEsp(Benchmark):
    """
    CodiEsp benchmark for ICD-10 coding (https://temu.bsc.es/codiesp).
    Original medical texts are in Spanish and have been translated with googletrans module into English.
    """

    def __init__(self, cfg: dict):
        self.name = "codiesp_en"
        super().__init__(name=self.name)
        self._cfg = cfg
        self.chunking = cfg["chunking"]
        self.id_columns = Benchmark.get_id_columns(self.chunking)
        self.data = BenchmarkData(cfg)

    def _read_text_files(
        self, text_dir: str, label_file: str, index: pd.Index
    ) -> pd.DataFrame:
        """
        text_dir    to read data from [ES]
        label_file  ground truth for diagnoses codes per text position
        """
        index_list, texts_list = [], []
        assert os.path.exists(text_dir)
        for f in [f for f in os.listdir(text_dir) if f.endswith(".txt")]:
            i = os.path.split(f)[-1].split(".")[0]
            assert not i.endswith("_en")  # ensure to read original ES texts
            if index is None or i in index:
                with open(os.path.join(text_dir, f), "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh.readlines()]
                    lines = [line for line in lines if len(line) > 2]
                    index_list.append(i)
                    texts_list.append(lines)
        df_texts = pd.DataFrame({"text": texts_list}, index=index_list)
        column_names = [
            "id",
            "diagnosis_procedure",
            "y_true",
            "description",
            "position [ES]",
        ]
        df_labels = pd.read_table(
            label_file, header=None, names=column_names, index_col=["id"]
        )
        if index is not None:
            df_labels = df_labels[df_labels.index.isin(index)]
        df_labels = (
            df_labels[["y_true", "description", "diagnosis_procedure"]]
            .groupby("id")
            .agg(list)
        )
        return df_texts.merge(df_labels, left_index=True, right_index=True)

    def _read_codes(self, file_name):
        with open(file_name, "r") as file:
            for code in file:
                yield code.strip()

    def _compute_line_lengths_from_original_reports(
        self, text_orig_dir, text_orig_file_fmt, pids: List
    ):
        # todo: move to helper fcts
        line_ends_list, line_ends_pred_list, line_numbers_list = [], [], []
        for pid in pids:
            fname = os.path.join(text_orig_dir, text_orig_file_fmt.format(pid))
            with open(fname, "r", encoding="utf-8") as f:
                line_ends, line_numbers = [], []
                line_end = 0
                text = f.read().splitlines()
                for i, line in enumerate(text):
                    line_end += len(line) + 1  # plus one for split symbol
                    line_numbers.append(i + 1)
                    line_ends.append(line_end)

            line_ends_list.append(line_ends)
            line_ends_pred_list.append([0] + line_ends[:-1])
            line_numbers_list.append(line_numbers)

        df = pd.DataFrame(
            {
                "report_id": pids,
                "line_end_pred": line_ends_pred_list,
                "line_end": line_ends_list,
                "line": line_numbers_list,
            }
        )
        df = df.explode(column=["line_end", "line_end_pred", "line"])
        df.line = df.line.astype(int)
        df.line_end = df.line_end.astype(int)
        df.line_end_pred = df.line_end_pred.astype(int)
        return df

    def merge_with_dataset_true(self, results: pd.DataFrame) -> pd.DataFrame:
        #  def _helper_merge_sentence(self, results:DataFrame) -> DataFrame:
        ## merge first assuming classification on sentence level
        assert self.chunking in ["sent", "noph"]
        results.reset_index(inplace=True)
        results_report_ids = results.report_id.unique()
        results.set_index(["report_id", "sentence_id"], inplace=True)

        # Restrict ground truth rows to report_ids in results
        dataset_true = self.get_dataset_true(
            dataset=self._cfg["benchmark"]["dataset"],
            report_ids=results_report_ids,
            with_lines=True,
        )

        # Aggregate y_true labels to lists and index on [report_id, sentence_id]
        dataset_true.rename(columns={"sentence": "sentence_id"}, inplace=True)
        dataset_true = (
            dataset_true[["report_id", "sentence_id", "y_true"]]
            .groupby(by=["report_id", "sentence_id"])
            .agg({"y_true": lambda c: c.tolist()}, axis=1)
        )

        # Finally merge results with ground truth on index
        results = results.merge(
            dataset_true, how="left", on=["report_id", "sentence_id"]
        )

        if self.chunking == "sent":
            return results

        # If level is np we have replicated index rows and aggregate them
        def ensure_list(x):
            if isinstance(x, str):
                try:
                    return eval(x)
                except:
                    return [x]
            return x if isinstance(x, list) else [x]

        results["y_pred"] = results["y_pred"].apply(ensure_list)
        results["y_true"] = results["y_true"].apply(ensure_list)

        # Group by index and aggregate
        results = results.groupby(["report_id", "sentence_id"]).agg(
            {
                "y_pred": lambda x: [
                    item for sublist in x for item in sublist
                ],  # flatten lists
                "y_true": lambda x: [
                    item for sublist in x for item in sublist
                ],  # flatten lists
                "dist": lambda x: [
                    item for sublist in x for item in sublist
                ],  # flatten lists
                "text": lambda x: x.tolist(),  # convert text column to list
            }
        )
        return results

    def get_description_text(self, code, include_ancestors: bool = False) -> str:
        """Note: no check here for is_leaf or is_category.
        Valid codes are returned by get_codes.

        """
        if include_ancestors:
            ancestor_line = lambda code: [code] + icd.get_ancestors(code)
            ancs = [code for code in ancestor_line(code)]
            if self._categories_only:
                ancs = [code for code in ancs if icd.is_category_or_subcategory(code)]
            return " ".join([icd.get_description(a) for a in ancs])
        return icd.get_description(code)

    def _recover_sentence_ids(
        self, data: pd.DataFrame, sentences: List[str], report_id: str
    ):
        # strategy: find all occurrences and build most flat monotonic series from sentence ids
        sentence_ids = [0]

        def iterate_over_sentence_ids(word):
            for j, sentence in enumerate(
                sentences[sentence_ids[-1] :], start=sentence_ids[-1]
            ):
                found_flag = False
                if word in sentence:
                    found_flag = True
                    break
            return j, found_flag

        for _, row in data.iterrows():
            if isinstance(row.text, float) and np.isnan(row.text):
                sentence_ids.append(sentence_ids[-1])
                continue
            j, found_flag = iterate_over_sentence_ids(row.text)
            if not found_flag and ". " in row.text:
                # search again with repaired noun phrase
                j, found_flag = iterate_over_sentence_ids(row.text.split(". ", 1)[-1])
            if not found_flag:
                raise LookupError(f"Could not find '{row.text}' in report {report_id}")
            sentence_ids.append(j)
        assert data.shape[0] + 1 == len(sentence_ids)
        return pd.Series(sentence_ids[1:])

    def get_reference_meta(self):
        return self.data.reference_index

    def get_reference_embeddings(self) -> np.array:
        return self.data.reference_e

    def get_test_meta(self) -> pd.DataFrame:
        return self.data.test_i
    
    def get_test_true(self) -> pd.DataFrame:
        columns = ['report_id', 'sentence_id', 'y_true']
        return self.data.test_true[columns]

    def get_test_embeddings(self) -> np.array:
        return self.data.test_e

    def get_train_meta(self) -> pd.DataFrame:
        return self.data.train_i
    
    def get_train_true(self) -> pd.DataFrame:
        columns = ['report_id', 'sentence_id', 'y_true']
        return self.data.train_true[columns]

    def get_train_embeddings(self) -> np.array:
        return self.data.train_e


