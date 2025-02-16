import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import re
from gensim.models import TfidfModel, nmf
from gensim.corpora import Dictionary as Dict
from gensim.models.ldamulticore import LdaModel
from multiprocessing import Pool
from functools import partial
from gensim.models.nmf_pgd import solve_h

CHUNK_SIZE = 1000
TOPIC_NUM = 100
ALLFILES = ['logon.csv','device.csv', 'email.csv', 'file.csv', 'http.csv']
CONTENT_FILES = ['email.csv', 'file.csv', 'http.csv']

def check():
    if not output_dir.is_dir():
        os.makedirs(output_dir)
    assert answers_dir.is_dir()
    assert dataset_dir.is_dir()
    assert main_answers_file.is_file()
    assert output_dir.is_dir()

def count_file_lines(file):
    with open(file) as f:
        for count, _ in enumerate(f):
            pass
    return count

def collect_vocabulary(csv_name):
    result_set = set()
    for df in pd.read_csv(dataset_dir / f'{csv_name}.csv',
                          usecols=['date', 'user', 'content'], chunksize=CHUNK_SIZE):
        df['content'] = df['content'].str.lower().str.split()
        result_set = result_set.union(*map(set, df['content']))
    return result_set

def chunk_iterator(filename, chunk_size=1000):
    """
    Generator that reads the CSV file in smaller chunks using the Python engine.
    It skips bad lines and yields tokenized documents from the 'content' column.
    """
    for chunk in pd.read_csv(
            filename,
            chunksize=chunk_size,
            engine='python',
            on_bad_lines='skip'  # Skips lines that cause errors
        ):
        if 'content' not in chunk.columns:
            continue
        # Tokenize the 'content' column: lower-case and split into words
        for document in chunk['content'].str.lower().str.split().values:
            yield document



def tfidf_iterator(filenames, dictionary):
    for filename in filenames:
        for chunk in pd.read_csv(dataset_dir / filename, chunksize=CHUNK_SIZE):
            for document in chunk['content'].str.lower().str.split().values:
                yield dictionary.doc2bow(document)

def nmf_iterator(filenames, dictionary, tfidf):
    for filename in filenames:
        for chunk in pd.read_csv(dataset_dir / filename, chunksize=CHUNK_SIZE):
            for document in chunk['content'].str.lower().str.split().values:
                yield tfidf[dictionary.doc2bow(document)]

# Parallelism
def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def process_content(filename, chunk_size=CHUNK_SIZE):
    model = LdaModel.load((output_dir / 'lda_model.pkl').as_posix())
    temp_dict = Dict.load((output_dir / 'dict.pkl').as_posix())
    out_file = output_dir / (filename.stem + '_lda.csv')
    if not out_file.is_file():
        Path(out_file).touch()
    for chunk in pd.read_csv(filename, usecols=['id', 'content'], chunksize=chunk_size):
        chunk['content'] = chunk['content'].str.lower().str.split() \
            .apply(lambda doc: model[temp_dict.doc2bow(doc)])
        chunk.to_csv(out_file, mode='a', index=False)

def pre_process_logon(path):
    df = pd.read_csv(path)
    # Convert the date column and create a day-of-week column
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S').dt.floor('D')
    df['day'] = df['date'].dt.dayofweek

    # Compute the most frequent PC used by each user on each day,
    # then group by user and take the mode (if multiple modes exist, take the first)
    self_pc = (df.groupby(['user', 'day', 'pc']).size().to_frame('count')
                  .reset_index().sort_values('count', ascending=False)
                  .drop_duplicates(subset=['user', 'day'])
                  .drop(columns=['count']).sort_values(['user', 'day'])
                  .groupby('user').pc.agg(lambda x: pd.Series.mode(x)[0])
                  .rename('self_pc'))
    # Convert self_pc to a DataFrame with an explicit 'user' column
    self_pc_df = self_pc.to_frame().reset_index()

    # Merge self_pc info into the main DataFrame using a left join on 'user'
    df = df.merge(self_pc_df, on='user', how='left')
    print("Logon merge done. Sample:")
    print(df.head())

    # Determine work time (8:00 to 17:00)
    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time

    # Use the 'activity' column as subtype if it exists; otherwise set to None
    df['subtype'] = df['activity'] if 'activity' in df.columns else None

    # Save the preprocessed logon data
    df[['id', 'date', 'user', 'is_work_time', 'subtype']].to_csv(output_dir / 'logon_preprocessed.csv', index=False)
    return self_pc_df

def pre_process_device(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')
    df = df.merge(self_pc, on='user', how='left')
    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time
    df['subtype'] = df['activity'] if 'activity' in df.columns else None
    df[['id', 'date', 'user', 'is_work_time', 'subtype']].to_csv(output_dir / 'device_preprocessed.csv', index=False)

def pre_process_file(path):
    df = pd.read_csv(path, usecols=['id', 'date', 'user', 'pc', 'filename'])
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')
    df = df.merge(self_pc, on='user', how='left')
    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time
    file_extensions = df.filename.str[-4:]
    df['subtype'] = file_extensions
    df[['id', 'date', 'user', 'is_work_time', 'subtype']].to_csv(output_dir / 'file_preprocessed.csv', index=False)

def pre_process_email(path):
    df = pd.read_csv(path, usecols=['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from'])
    df = df.fillna('')
    to_concated = df[['to', 'cc', 'bcc']].apply(lambda x: ';'.join([x.to, x.cc, x.bcc]), axis=1)
    is_external_to = to_concated.apply(
        lambda x: any([re.match('^.+@(.+$)', e).group(1) != 'dtaa.com' for e in x.split(';') if e != '']))
    is_external = is_external_to  # or combine with additional logic if needed
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')
    df = df.merge(self_pc, on='user', how='left')
    df['is_usual_pc'] = df['self_pc'] == df['pc']
    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time
    df['subtype'] = is_external
    df[['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype']].to_csv(output_dir / 'email_preprocessed.csv', index=False)

def pre_process_http(path):
    # Scenario definitions
    scenario_1_http = [
        'actualkeylogger.com',
        'best-spy-soft.com',
        'dailykeylogger.com',
        'keylogpc.com',
        'refog.com',
        'relytec.com',
        'softactivity.com',
        'spectorsoft.com',
        'webwatchernow.com',
        'wellresearchedreviews.com',
        'wikileaks.org'
    ]
    scenario_2_http = [
        'careerbuilder.com',
        'craiglist.org',
        'indeed.com',
        'job-hunt.org',
        'jobhuntersbible.com',
        'linkedin.com',
        'monster.com',
        'simplyhired.com',
    ]
    scenario_3_http = [
        '4shared.com',
        'dropbox.com',
        'fileserve.com',
        'filefreak.com',
        'filestube.com',
        'megaupload.com',
        'thepiratebay.org'
    ]
    
    first_it = True
    mode = 'w'
    # Optionally reduce CHUNK_SIZE if memory remains an issue:
    local_chunk_size = 500  # Adjust this value as needed

    for http_df in pd.read_csv(
            path,
            chunksize=local_chunk_size,   # Use a smaller chunk size to reduce memory load
            usecols=['id', 'date', 'user', 'pc', 'url'],
            engine='python',              # Use the Python engine
            on_bad_lines='skip'           # Skip lines that cause parsing errors (for pandas >= 1.3.0)
        ):
        # Convert the 'date' column to datetime
        http_df['date'] = pd.to_datetime(http_df.date, format='%m/%d/%Y %H:%M:%S', errors='coerce')
        
        # Extract site names using a regex; handle missing matches gracefully
        http_df['site_name'] = http_df['url'].apply(
            lambda s: re.match(r'^https?://(www\.)?([0-9\-\w\.]+)?.+$', s).group(2) if re.match(r'^https?://(www\.)?([0-9\-\w\.]+)?.+$', s) else ''
        )
        
        # Initialize subtype column and assign values based on scenarios
        http_df['subtype'] = 0
        http_df.loc[http_df['site_name'].isin(scenario_1_http), 'subtype'] = 1
        http_df.loc[http_df['site_name'].isin(scenario_2_http), 'subtype'] = 2
        http_df.loc[http_df['site_name'].isin(scenario_3_http), 'subtype'] = 3
        
        # Merge with self_pc DataFrame (ensure self_pc is available globally)
        http_df = http_df.merge(self_pc, on='user', how='left')
        
        # Determine work time: between 8:00 and 17:00
        is_work_time = (8 <= http_df['date'].dt.hour) & (http_df['date'].dt.hour < 17)
        http_df['is_work_time'] = is_work_time
        
        # Write the processed chunk to CSV
        http_df.to_csv(
            output_dir / 'http_preprocessed.csv',
            header=first_it,
            index=False,
            mode=mode,
            columns=['id', 'date', 'user', 'is_work_time', 'subtype', 'site_name']
        )
        first_it = False
        mode = 'a'


def merge_all_content():
    # Create a dictionary from the 'email.csv' file using the modified iterator
    df_dict = Dict(chunk_iterator(dataset_dir / 'email.csv', chunk_size=1000))
    df_dict.add_documents(chunk_iterator(dataset_dir / 'file.csv', chunk_size=1000))
    df_dict.add_documents(chunk_iterator(dataset_dir / 'http.csv', chunk_size=1000))
    df_dict.save((output_dir / 'dict.pkl').as_posix())


def make_tfidf_model():
    tfidf_model = TfidfModel(
        tfidf_iterator(CONTENT_FILES, Dict.load((output_dir / 'dict.pkl').as_posix())))
    tfidf_model.save((output_dir / 'tfidf_model.pkl').as_posix())

def make_nmf_model():
    tfidf_model = TfidfModel.load((output_dir / 'tfidf_model.pkl').as_posix())
    nmf_model = nmf.Nmf(
        nmf_iterator(CONTENT_FILES, Dict.load((output_dir / 'dict.pkl').as_posix()),
                     tfidf_model), num_topics=TOPIC_NUM)
    nmf_model.save((output_dir / 'nmf_model.pkl').as_posix())

def make_lda_model():
    tfidf_model = TfidfModel.load((output_dir / 'tfidf_model.pkl').as_posix())
    lda_model = LdaModel(
        nmf_iterator(CONTENT_FILES, Dict.load((output_dir / 'dict.pkl').as_posix()),
                     tfidf_model), num_topics=TOPIC_NUM)
    lda_model.save((output_dir / 'lda_model.pkl').as_posix())

if __name__ == "__main__":
    # Define directories
    answers_dir = Path("./answers")
    dataset_dir = Path("./dataset")
    main_answers_file = answers_dir / "insiders.csv"
    output_dir = Path('./_output/')

    if not output_dir.is_dir():
        os.mkdir(output_dir)

    # Preprocess logon data and compute self_pc
    self_pc = pre_process_logon(dataset_dir / 'logon.csv')
    print("Logon processed")

    pre_process_device(dataset_dir / 'device.csv')
    print("Device processed")

    pre_process_file(dataset_dir / 'file.csv')
    print("File processed")

    #pre_process_email(dataset_dir / 'email.csv')
    #print("Email processed")

    pre_process_http(dataset_dir / 'http.csv')
    print("HTTP processed")

    merge_all_content()
    print("All content merged and saved")
    make_tfidf_model()
    print("TF-IDF model saved")
    make_nmf_model()
    print("NMF model saved")
    make_lda_model()
    print("LDA model saved")

    for file in CONTENT_FILES:
        process_content(dataset_dir / file)
        print(file, "content processed")
