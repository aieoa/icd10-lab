import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import simple_icd_10_cm as icd
from typing import List
import sys


sys.path.append("..")

from src.benchmarks.base import Benchmark, BenchmarkFactory
from src.benchmarks.data import BenchmarkData
from src.utils.nlp import split_into_sentences
from src.utils.sysops import get_repo_root

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

    def get_reference_set(self):
        return self.data.reference_index

    def get_reference_embeddings(self):
        return self.data.reference_e

    def get_test_set(self):
        return self.data.test_i

    def get_test_embeddings(self, device: str = None):
        return self.data.test_e

    def get_train_set(self) -> pd.DataFrame:
        return self.data.train_i

    def get_train_embeddings(self) -> pd.DataFrame:
        return self.data.train_e


def calculate_sentence_boundaries(data_dir, split):
    sentence_boundaries = {}
    sentence_texts = {}
    text_orig_dir = data_dir / "01_raw" / "codiesp_es" / split / "text_files"
    for fname in text_orig_dir.glob("*.txt"):
        report_id = fname.stem
        with open(os.path.join(text_orig_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        sentences = split_into_sentences(text, language="es")
        acc = [
            sum(len(s) for s in sentences[: i + 1]) - 1 for i in range(len(sentences))
        ]
        sentence_boundaries[report_id] = acc
        sentence_texts[report_id] = sentences
    return sentence_boundaries, sentence_texts


def lookup_sentence_id(sentence_boundaries, report_id, pos_end):
    import bisect

    acc = sentence_boundaries[report_id]
    idx = bisect.bisect_left(acc, pos_end)
    if idx >= len(acc):
        idx = len(acc) - 1  # clamp to final sentence index
    return idx + 1  # convert to 1-based


def _code_cleanser(code):
    import re

    code_rx = re.compile(r"(?P<code>([A-Z]\d+\.\d+|[A-Z]\d+))[\.A-Z|A-Z]+")
    try:
        icd.get_description(code)
    except ValueError:
        mobj = code_rx.match(code)
        if mobj:
            code_new = mobj.group("code")
            try:
                icd.get_description(code_new)
            except ValueError:
                raise ValueError(f"ERROR\t{code_new} also does not exist")
            return code_new
        else:
            return ""
    return code


def extract_annotations_from_split(data_dir, split, ground_truth_file):
    # sentence_boundaries = calculate_sentence_boundaries(data_dir, split)
    def get_es_sentence(row):
        sents = sentence_texts.get(row["report_id"], [])
        idx = row["sentence_id"] - 1  # Convert 1-based to 0-based
        return sents[idx] if 0 <= idx < len(sents) else ""

    sentence_boundaries, sentence_texts = calculate_sentence_boundaries(data_dir, split)

    df = pd.read_csv(ground_truth_file, sep="\t", encoding="utf-8", header=None)
    df.rename(
        columns={
            0: "report_id",
            1: "diagnose_procedure",
            2: "code",
            3: "description",
            4: "positions",
        },
        inplace=True,
    )
    df = df[df.diagnose_procedure == "DIAGNOSTICO"]
    df.positions = df.positions.apply(lambda x: x.split(";"))
    df = df.explode(column="positions")
    df["pos_end"] = df.positions.apply(lambda x: int(x.split(" ")[-1]))
    df.code = df.code.str.upper()
    df.code = df.code.apply(_code_cleanser)
    df = df[df.code != ""]
    assert not df.empty
    for idx, row in df.iterrows():
        s_id = lookup_sentence_id(sentence_boundaries, row["report_id"], row["pos_end"])
        if s_id > len(sentence_boundaries[row["report_id"]]):
            print(
                f"Warning: Annotation pos_end={row['pos_end']} exceeds detected sentence boundary in report {row['report_id']}"
            )

    sentence_ids = df.apply(
        lambda row: lookup_sentence_id(
            sentence_boundaries, row["report_id"], row["pos_end"]
        ),
        axis=1,
    )

    df.insert(1, "sentence_id", sentence_ids)
    df = df.sort_values(by="report_id").rename(columns={"code": "y_true"})
    df = df.drop(columns=["diagnose_procedure", "positions", "pos_end"])
    df["sentence_es"] = df.apply(get_es_sentence, axis=1)

    return df.dropna()


def collect_sentences(src_dir: Path, split):
    df_list = [pd.read_csv(f) for f in src_dir.glob("*.csv")]
    for f in src_dir.glob("*.csv"):
        df = pd.read_csv(f)
        assert df.shape[1] == 3, df.head(2)
    assert len(df_list) == BENCHMARK_FILE_CTRS[split]
    big_df = pd.concat(df_list, ignore_index=True)
    big_df = big_df.rename(columns={"sentence": "sentence_en"})
    print(big_df.head())
    return big_df


def collect_annotations(data_dir: Path):
    # create y_true sets for train and test splits and persist to memory
    tgt_dir = data_dir / "02_processed" / "codiesp_en"
    tgt_dir.mkdir(parents=True, exist_ok=True)
    src_dir = data_dir / "01_raw" / "codiesp_es"
    assert src_dir.exists()
    for split in ["test", "train"]:
        tgt_path = tgt_dir / split / "y_true.csv"
        ground_truth_file = src_dir / split / f"{split}X.tsv"
        assert os.path.exists(ground_truth_file)
        df_es = extract_annotations_from_split(data_dir, split, ground_truth_file)

        ## merge on report_id, sentence_id with sentence chunked texts under
        # /data/02_processed/codiesp_en/[test|train]/sent/{report_id}.csv
        df_en = collect_sentences(tgt_path.parent / "sent", split)

        merged = pd.merge(
            left=df_en,
            right=df_es,
            on=["report_id", "sentence_id"],
            how="left",
            indicator=True,
        )

        # check ICD10 rows (df_es) without an English sentence (df_en) match
        es_no_match = merged[merged["_merge"] == "right_only"]

        assert len(es_no_match) == 0, es_no_match

        merged.drop(columns=["_merge"], inplace=True)
        deduped = merged.drop_duplicates(subset=["report_id", "sentence_id", "y_true"])
        deduped.to_csv(tgt_path, index=False)
        print(f"INFO\tGround truth for {split} split written to {tgt_path}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--collect-annotations",
    action="store_true",
    help="Collect and persist to memory y true sets",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default=Path(get_repo_root()) / "data",
    help="Data base dir",
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.collect_annotations:
        collect_annotations(Path(args.data_dir))
