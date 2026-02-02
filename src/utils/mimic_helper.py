import argparse
import os
import pandas as pd

TIME_COLUMNS = ['admittime', 'dischtime']

# diagnoses + admissions merge stuff

def load_mimic_iv_admissions_with_icd_codes(data_dir: str):
    '''
    Load and merge diagnostics and admissions tables. Merge is performed
    according to the specification in the MIMIC-IV documentation
    
    :param data_dir: Description
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
    #main()