# icd10-lab/src/utils/nlp.py


def split_into_sentences(text):
    from sentence_splitter import SentenceSplitter

    splitter = SentenceSplitter(language="en")
    sentences = [s.replace("\n", "") for s in splitter.split(text)]
    return [s for s in sentences if len(s) > 2]


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
