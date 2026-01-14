import argparse
import os
import pandas as pd
from pathlib import Path
import simple_icd_10_cm as icd
import sys


sys.path.append("..")

from src.utils.nlp import split_into_sentences
from src.utils.sysops import get_repo_root

BENCHMARK_FILE_CTRS = {"train": 500, "test": 250}


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
