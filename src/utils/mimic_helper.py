import argparse
import os
import pandas as pd

TIME_COLUMNS = ['admittime', 'dischtime']


# notes merge with processed admissions
def load_and_merge_notes(data_dir: dir, adm_dia: pd.DataFrame) -> pd.DataFrame:
    """
    load notes from data directory, process it and merge with admissions dataframe
    
    :param data_dir: data directory with "mimic" folder inside
    :type data_dir: dir
    :param adm_dia: admissions dataframe with diagnostic codes added and aggregated
    :type adm_dia: pd.DataFrame
    :return: Merged dataframe with notes and diagnoses
    :rtype: DataFrame
    """

    notes = pd.read_csv(data_dir + "/mimic/" + "discharge.csv.gz")
    filtered_notes = filter_notes(notes)
    merged_df = pd.merge(adm_dia, filtered_notes, on=['hadm_id'], how='left')

    return merged_df


# notes cleaner + filterer 
def filter_notes(notes_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # strip texts from leading and trailing and white spaces
    notes_df["text"] = notes_df["text"].str.strip()

    # remove entries without subject id or text
    notes_df = notes_df.dropna(subset=["subject_id", "text"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "chief_complaint": "chief complaint:",
        "present_illness": "present illness:",
        "medical_history": "medical history:",
        "medication_adm": "medications on admission:",
        "allergies": "allergies:",
        "physical_exam": "physical exam:",
        "family_history": "family history:",
        "social_history": "social history:"
    }

    # replace linebreak indicators
    # notes_df['text'] = notes_df['text'].str.replace("\n", "\\n")
    notes_df['text'] = notes_df['text'].str.replace("___\nFamily History:", "___\n\nFamily History:", flags=re.IGNORECASE)

    # extract each section by regex
    for key, section in admission_sections.items():
        notes_df[key] = notes_df.text.str.extract('{}([\\s\\S]+?)\n\\s*?\n[^(\\\\|\\d|\\.)]+?:'.format(section), flags=re.IGNORECASE)

        notes_df[key] = notes_df[key].str.replace('\n', ' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df.loc[notes_df[key].str.startswith("[]"), key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.chief_complaint != "") | (notes_df.present_illness != "") |
                        (notes_df.medical_history != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(text="CHIEF COMPLAINT: " + notes_df.chief_complaint.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.present_illness.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.medical_history.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.medication_adm.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.allergies.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.physical_exam.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.family_history.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.social_history.astype(str))['text']

    return notes_df


# diagnoses + admissions merge stuff

def load_mimic_iv_admissions_with_icd_codes(data_dir: str):
    '''
    Load and merge diagnostics and admissions tables. Merge is performed
    according to the specification in the MIMIC-IV documentation
    
    :param data_dir: Data directory with "mimic" folder inside
    :type data_dir: str
    '''
    mimiciv_dir = data_dir + '/mimic'

    ''' for reference remove later
    CONTAINS 
    ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
       'admission_type', 'admit_provider_id', 'admission_location',
       'discharge_location', 'insurance', 'language', 'marital_status', 'race',
       'edregtime', 'edouttime', 'hospital_expire_flag']'''
    adm = pd.read_csv(mimiciv_dir + '/admissions.csv.gz', parse_dates=TIME_COLUMNS)

    ''' for reference, remove later
    CONTAINS
    ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']'''
    dia = pd.read_csv(mimiciv_dir + "/diagnoses_icd.csv.gz")

    #removing icd-9 codes as they are not needed
    dia = dia[dia['icd_version'] == 10]
    dia = dia.drop(columns='icd_version')
    for df in [adm, dia]:
        df.rename(str.lower, axis=1, inplace=True)

    return add_required_columns(adm, dia)

def add_required_columns(admissions: pd.DataFrame, dia_codes: pd.DataFrame):
    admissions = add_codes_to_adm(admissions, dia_codes, 'dia')
    times_to_moment(admissions, *TIME_COLUMNS)
    return admissions

def times_to_moment(df: pd.DataFrame, *times: str):
    for t in times:
        df[t.replace('time', 'moment')] = df[t].dt.time
        if t != 'admittime':
            df[t.replace('time', 'interval')] = (df[t] - df['admittime']).dt.total_seconds()


def add_codes_to_adm(admissions: pd.DataFrame, codes: pd.DataFrame, code_type: str):
    combined_codes = codes \
        .dropna() \
        .groupby(['hadm_id', 'subject_id']) \
        .agg(set) \
        .rename({'icd_code': code_type}, axis=1) \
        .reset_index()
    adm_with_codes = pd.merge(admissions, combined_codes, on=['hadm_id', 'subject_id'], how='left').reset_index()
    nans = adm_with_codes[code_type].isna()
    adm_with_codes.loc[nans, code_type] = adm_with_codes[code_type][nans].apply(lambda x: set())
    return adm_with_codes.set_index(['hadm_id', 'subject_id'])


def main():
    parser = argparse.ArgumentParser(description="Process data in a directory")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data_dir)"
    )
    parser.add_argument(
        "--limit",
        "--l",
        type=int,
        default=None,
        help="No of rows to read from the dataset (default: read the entire csv files)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"

    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {output_dir}")



if __name__ == "__main__":
    data = load_mimic_iv_admissions_with_icd_codes('data')
    data_with_notes = load_and_merge_notes('data', data)
