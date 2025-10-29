# icd10-lab/src/utils/nlp.py


def split_into_sentences(text, language="en"):
    import re
    from sentence_splitter import SentenceSplitter

    ABBREVIATIONS = ["i.v.", "Sr.", "Dr."]

    def protect_abbr(text, abbr_list):
        for abbr in abbr_list:
            text = text.replace(abbr, abbr.replace(".", "_dot_"))
        return text

    def restore_abbr(text, abbr_list):
        for abbr in abbr_list:
            text = text.replace(abbr.replace(".", "_dot_"), abbr)
        return text

    splitter = SentenceSplitter(language=language)
    text = protect_abbr(text, ABBREVIATIONS)

    sentences = [s.replace("\n", "") for s in splitter.split(text)]
    sentences = [s for s in sentences if len(s) > 2]
    resplit = []
    # Only apply the pT4G3N1 rule in English context
    pT4G3N1_rule = r"(?<!^)(?<![.])(?=\bpT4G3N1\b)" if language == "en" else ""
    main_re = r"(?<=\.)\s+(?=[A-Z])" r"|(?<=cm\.)\s+" r"|(?<=\.\s)\("
    full_regex = main_re
    if pT4G3N1_rule:
        full_regex += "|" + pT4G3N1_rule

    for s in sentences:
        secondary = re.split(full_regex, s)
        resplit.extend([q.strip() for q in secondary if len(q.strip()) > 2])
    return [restore_abbr(s, ABBREVIATIONS) for s in resplit]


def split_into_noun_phrases(text):
    import spacy

    # if missing, download with `python -m spacy download en_core_web_sm`
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def split_into_noun_phrases_with_verbs(text):
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    results = []
    for chunk in doc.noun_chunks:
        # Find verb immediately following the chunk
        right_verb = None
        end_i = chunk.end
        if end_i < len(doc) and doc[end_i].pos_ == "VERB":
            right_verb = doc[end_i].text
        if right_verb:
            span = f"{chunk.text} {right_verb}"
        else:
            span = chunk.text
        results.append(span)
    return results


def clean_icd10_code(code: str):
    import re
    import simple_icd_10_cm as icd

    code_rx = re.compile(r"(?P<code>([A-Z]\d+\.\d+|[A-Z]\d+))[\.A-Z|A-Z]+")
    try:
        icd.get_description(code)
    except ValueError:
        # print(f'ERROR\t{code} does not exists')
        mobj = code_rx.match(code)
        if mobj:
            code_new = mobj.group("code")
            try:
                icd.get_description(code)
            except ValueError:
                # print(f'ERROR\t{code_new} also does not exist')
                return ""
            return code_new
        else:
            return ""
    return code
