# icd10-lab/tests/utils/test_nlp.py

import os
import pytest
from unittest.mock import patch
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


from src.utils.nlp import *


@pytest.fixture
def default_sentence():
    sentence = (
        "The histological examination was: 21 cm of transverse colon with "
        "measuring 7 cm in extension, areas of linear ulceration with "
        "inflammatory infiltrate with microabscesses affecting the entire "
        "thickness of the specimen. Two months later the gastrin value was "
        "529 pg/ml."
    )
    return sentence


@pytest.fixture
def default_text():
    sentence = (
        "The patient did not agree to undergo new examinations and "
        "maintained good general condition for 2 and a half years, having "
        "one or two stools a day formed with some anal incontinence, he "
        "received iron for iron deficiency anemia, plantago ovata and "
        "omeprazole 20 mg per day. In October 2004, a colonoscopy was "
        "performed for diarrhea, which showed segmental colitis from 15 cm "
        "of the anal verge to 35 cm with pseudopolyps and friability that "
        "appeared to be ulcerative colitis. The rectum did not have lesions. "
        "Colon biopsies suggested active ulcerative colitis. The patient was "
        "treated with oral mesalazine and then steroids from November 2004 "
        "to February 2005. The patient was admitted several times for "
        "worsening pulmonary emphysema and suffered a bilateral "
        "pneumothorax. He was followed only in pulmonology consultations "
        "until April 2006."
    )
    return sentence


def test_call_split_into_noun_phrases(default_sentence):
    chunks = split_into_noun_phrases(default_sentence)
    print(f"chunks in np wo verbs: {len(chunks)}, {chunks}")
    assert len(chunks) == 14


def test_call_split_into_noun_phrases_with_verbs(default_sentence):
    chunks = split_into_noun_phrases_with_verbs(default_sentence)
    print(f"chunks in np w verbs: {len(chunks)}, {chunks}")
    assert len(chunks) == 14


def test_split_into_sentences(default_text):
    sentences = split_into_sentences(default_text)
    assert len(sentences) == 7
