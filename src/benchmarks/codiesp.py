import numpy as np
import os
import pandas as pd
import re
import simple_icd_10_cm as icd
import torch
from typing import List


from src.benchmarks.base import Benchmark, BenchmarkFactory
from src.benchmarks.data import BenchmarkData
import src.utils.nlp as nlp
import src.utils.sysops as sysops

@BenchmarkFactory.register("CodiEsp")
class CodiEsp(Benchmark):
    """
    CodiEsp benchmark for ICD-10 coding (https://temu.bsc.es/codiesp).
    Original medical texts are in Spanish and have been translated with googletrans module into English.
    """
    def __init__(self, cfg:dict):
        self.name = 'codiesp'
        super().__init__(name=self.name)
        self._cfg = cfg
    
        data_dir = cfg['data_dir']
        self.chunking = cfg['chunking']
        self.id_columns = Benchmark.get_id_columns(self.chunking)
        self.data = BenchmarkData(cfg)
        self.test_index = self._cfg['subset'].get('test', [])
        self.train_index = self._cfg['subset'].get('train', [])
        self.test_index = self.test_index if self.test_index and len(self.test_index) else None
        self.train_index = self.train_index if self.train_index and len(self.train_index) else None
         

        self.test_true_file = os.path.join(data_dir, 'final_dataset_v4_to_publish', 'test', 'testX.tsv') 
        self.train_true_file = os.path.join(data_dir, 'final_dataset_v4_to_publish', 'train', 'trainX.tsv')
        assert os.path.exists(self.test_true_file), f"test true file does not exist: {self.test_true_file}"
        assert os.path.exists(self.train_true_file), f"train true file does not exist: {self.train_true_file}"
        
        # path to original text files
        self.test_text_orig = os.path.join(data_dir, 'final_dataset_v4_to_publish', 'test', 'text_files')
        self.test_text_orig_file_fmt = '{}.txt'

        self.train_text_orig = os.path.join(data_dir, 'final_dataset_v4_to_publish', 'train', 'text_files')
        self.train_text_orig_file_fmt = '{}.txt'

        assert os.path.exists(self.test_text_orig)
        assert os.path.exists(self.train_text_orig)
        
        self._reference_e_fmt = 'last_layer_{}.pt'
        self._reference_i_fmt = 'last_layer_{}.pt.idx'

        # test or train set embedding patterns
        self._set_e_rx = re.compile(r'^(?P<report_id>.+)_en.pt$')
        self._set_i_fmt = '{}_en.pt.idx'


    def _read_text_files(self, text_dir:str, label_file:str, index:pd.Index) -> pd.DataFrame:
        """
            text_dir    to read data from [ES]
            label_file  ground truth for diagnoses codes per text position
        """
        index_list, texts_list = [], []
        assert os.path.exists(text_dir)
        for f in [f for f in os.listdir(text_dir) if f.endswith('.txt')]:
            i = os.path.split(f)[-1].split('.')[0]
            assert not i.endswith('_en')  # ensure to read original ES texts
            if index is None or i in index:
                with open(os.path.join(text_dir, f), 'r', encoding='utf-8') as fh: 
                    lines = [line.strip() for line in fh.readlines()]
                    lines = [line for line in lines if len(line) > 2]
                    index_list.append(i)
                    texts_list.append(lines)
        df_texts = pd.DataFrame({'text': texts_list}, index=index_list)
        column_names = ['id', 'diagnosis_procedure', 'y_true', 'description', 'position [ES]']
        df_labels = pd.read_table(label_file, header=None, names=column_names, index_col=['id'])
        if index is not None:
            df_labels = df_labels[df_labels.index.isin(index)]
        df_labels = df_labels[['y_true', 'description', 'diagnosis_procedure']].groupby('id').agg(list)
        return df_texts.merge(df_labels, left_index=True, right_index=True)
    
    def _read_codes(self, file_name):
        with open(file_name, 'r') as file:
            for code in file:
                yield code.strip()


    def _compute_line_lengths_from_original_reports(self, text_orig_dir, text_orig_file_fmt, pids:List):
        # todo: move to helper fcts
        line_ends_list, line_ends_pred_list, line_numbers_list = [], [], []
        for pid in pids:
            fname = os.path.join(text_orig_dir, text_orig_file_fmt.format(pid))
            with open(fname, 'r', encoding='utf-8') as f:
                line_ends, line_numbers = [], []
                line_end = 0
                text = f.read().splitlines()
                for i, line in enumerate(text):
                    line_end += len(line) + 1 # plus one for split symbol
                    line_numbers.append(i + 1)
                    line_ends.append(line_end)

            line_ends_list.append(line_ends)
            line_ends_pred_list.append([0] + line_ends[:-1])
            line_numbers_list.append(line_numbers)

        df = pd.DataFrame({'report_id': pids, 'line_end_pred': line_ends_pred_list, 'line_end': line_ends_list, 'line': line_numbers_list})
        df = df.explode(column=['line_end', 'line_end_pred', 'line'])
        df.line = df.line.astype(int)
        df.line_end = df.line_end.astype(int)
        df.line_end_pred = df.line_end_pred.astype(int)
        return df
    
    
    def _get_sentence_id(self, dataset:str, report_id:str, pos_end:int) -> int:
        import bisect
        import numpy as np
        from sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter(language='en')
        text_orig_file_fmt = eval(f'self.{dataset}_text_orig_file_fmt')
        text_orig = eval(f'self.{dataset}_text_orig')
        
        fname = os.path.join(text_orig, text_orig_file_fmt.format(report_id))
        with open(fname, 'r', encoding='utf-8') as f:
            text = f.read()
        assert len(text) > 1
        sentences = splitter.split(text)  #[s.replace('\n', ' ') for s in splitter.split(text)]
       
        acc = [a - 1 for a in np.cumsum([len(s) for s in sentences])]
        return bisect.bisect_left(acc, x=pos_end)
    
    
    def get_dataset_true(self, dataset:str, report_ids:List, with_lines:bool=False):
        # todo: move to notebook and persist retrieved ground truth datasets
        """ augment optionally with line information
            dataset:  'train' or 'test'
        
        """
        assert len(report_ids)
        ground_truth_file = eval(f'self.{dataset}_true_file')
        text_orig = eval(f'self.{dataset}_text_orig')
        text_orig_file_fmt = eval(f'self.{dataset}_text_orig_file_fmt')
        # ground_truth_file:str, text_orig:str, text_orig_file_fmt:str, 
        
        df = pd.read_csv(ground_truth_file, sep='\t', encoding = "utf-8", header=None)
        df.rename(columns={0: 'report_id', 1: 'diagnose_procedure', 2: 'code', 3: 'description', 4: 'positions'}, inplace=True)
        df = df[df.report_id.isin(report_ids)]
        df = df[df.diagnose_procedure == 'DIAGNOSTICO']
        df.positions = df.positions.apply(lambda x: x.split(';'))
        df = df.explode(column='positions')
        df['pos_end'] = df.positions.apply(lambda x: int(x.split(' ')[-1]))
        if with_lines:
            orig_lines_df = self._compute_line_lengths_from_original_reports(text_orig, text_orig_file_fmt, report_ids)
            df = df.merge(orig_lines_df, on='report_id')
            in_range = (df.pos_end <= df.line_end) & (df.pos_end > df.line_end_pred)
            df = df[in_range]
            df.sort_values('line', inplace=True)
        df.code = df.code.str.upper()
        # remove refinements if they do not exist in icd10
        n_df = df.shape[0]
        df.code = df.code.apply(self._code_cleanser)
        df = df[df.code != '']
        assert not df.empty
        df['sentence'] = df.apply(lambda row: 
                            self._get_sentence_id(
                                dataset=dataset, 
                                report_id=row['report_id'], 
                                pos_end=row['pos_end']
                            ), axis=1)
        
        print(f'INFO\tDropped {n_df - df.shape[0]} rows')
        df = df.sort_values(by='report_id').rename(columns={'code': 'y_true'})
        df_aux = df[df.y_true.apply(lambda code: icd.is_leaf(code))]
        if self._cfg['benchmark'].get('keep_leaves_only', True):
            print(f'INFO\tThere are {df.shape[0] - df_aux.shape[0]} non-leaf codes and there are DROPPED (keep_leaves_only=True)')
            df = df_aux 
        else: 
            print(f'INFO\tThere are {df.shape[0] - df_aux.shape[0]} non-leaf codes and there are NOT DROPPED (keep_leaves_only=False)')
            
        # df['y_true'] = df['y_true'].apply(lambda code: code if icd.is_leaf(code) else pd.NA)
        return df.dropna()
    
    def merge_with_dataset_true(self, results:pd.DataFrame) -> pd.DataFrame:
        #  def _helper_merge_sentence(self, results:DataFrame) -> DataFrame:
        ## merge first assuming classification on sentence level
        assert self.chunking in ['sent', 'noph']
        results.reset_index(inplace=True)
        results_report_ids = results.report_id.unique()
        results.set_index(['report_id', 'sentence_id'], inplace=True)
        
        # Restrict ground truth rows to report_ids in results
        dataset_true = self.get_dataset_true(
            dataset=self._cfg['benchmark']['dataset'], 
            report_ids=results_report_ids, 
            with_lines=True)
        
        # Aggregate y_true labels to lists and index on [report_id, sentence_id]
        dataset_true.rename(columns={'sentence': 'sentence_id'}, inplace=True)
        dataset_true = dataset_true[['report_id', 'sentence_id', 'y_true']].\
            groupby(by=['report_id', 'sentence_id']).agg({'y_true': lambda c: c.tolist()}, axis=1)
        
        # Finally merge results with ground truth on index
        results = results.merge(dataset_true, how='left', on=['report_id', 'sentence_id'])

        if self.chunking == 'sent':
            return results
        
        # If level is np we have replicated index rows and aggregate them
        def ensure_list(x):
            if isinstance(x, str):
                try:
                    return eval(x)
                except:
                    return [x]
            return x if isinstance(x, list) else [x]
        results['y_pred'] = results['y_pred'].apply(ensure_list)
        results['y_true'] = results['y_true'].apply(ensure_list)
        
        # Group by index and aggregate
        results = results.groupby(['report_id', 'sentence_id']).agg({
            'y_pred': lambda x: [item for sublist in x for item in sublist],  # flatten lists
            'y_true': lambda x: [item for sublist in x for item in sublist],  # flatten lists
            'dist': lambda x: [item for sublist in x for item in sublist],  # flatten lists
            'text': lambda x: x.tolist()  # convert text column to list
        })
        return results 
    
    
    def get_description_text(self, code, include_ancestors:bool=False) -> str:
        """ Note: no check here for is_leaf or is_category. 
                Valid codes are returned by get_codes.
        
        """
        if include_ancestors:
            ancestor_line = lambda code: [code] + icd.get_ancestors(code)
            ancs = [code for code in ancestor_line(code)]
            if self._categories_only:
                ancs = [code for code in ancs if icd.is_category_or_subcategory(code)]
            return ' '.join([icd.get_description(a) for a in ancs])
        return icd.get_description(code)
        

    def _recover_sentence_ids(self, data:pd.DataFrame, sentences:List[str], report_id:str):
        # strategy: find all occurrences and build most flat monotonic series from sentence ids
        sentence_ids = [0]
        
        def iterate_over_sentence_ids(word):
            for j, sentence in enumerate(sentences[sentence_ids[-1]:], start=sentence_ids[-1]):
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
            if not found_flag and '. ' in row.text:
                # search again with repaired noun phrase
                j, found_flag = iterate_over_sentence_ids(row.text.split('. ', 1)[-1])
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
    
    def get_test_embeddings(self, device:str=None):
        return self.data.test_e
    
    def get_train_set(self) -> pd.DataFrame:
        return self.data.train_i
    
    # TODO: add index argument possibly
    def get_train_embeddings(self) -> pd.DataFrame:
        return self.data.train_e
    
    