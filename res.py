import re
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd
import os
from model import Model
from langdetect import detect

model = Model('english-partut-ud-2.0-170801.udpipe')
def create_df_from_conllu(conllu):
    conllu = re.sub('# .+?\n', '', conllu)
    l = conllu.split('\n')
    string = 'Id\tForm\tLemma\tUPosTag\tXPosTag\tFeats\tHead\tDepRel\tDeps\tMisc\n'
    string += "\n".join(l)
    TESTDATA = StringIO(string)
    df = pd.read_csv(TESTDATA, sep="\t")
    return df


def delete_from_df(df):
    to_drop = []
    for index, row in df.iterrows():
        if '-' in str(row['Id']):
            num = len(row['Id'].split('-'))
            for x in range(num):
                if num != 0:
                    to_drop.append(index + num)
                    num -= 1
    return to_drop


def parsing(text):
    sentences = model.tokenize(text)
    for s in sentences:
        model.tag(s)
        model.parse(s)
    conllu = model.write(sentences, "conllu")
    return conllu


def open_ann_find_end(ann):
    ts = re.findall('\n(T[0-9]+)\t', ann)
    ts = sorted([int(t[1:]) for t in ts])
    grids = re.findall('\n(#[0-9]+)\t', ann)
    grids = sorted([int(grid[1:]) for grid in grids])
    if grids == [] and ts != []:
        return ts[-1]+1, 1
    elif grids != [] and ts == []:
        return 1, grids[-1]+1
    elif grids == [] and ts == []:
        return 1, 1
    else:
        return ts[-1]+1, grids[-1]+1
    
    
def opening(path):
    with open(path, 'r') as f:
        text = f.read()
    if '\ufeff' in text:
        text = re.sub('\n', '\r\n', text)
    return text


def df_changing(text):
    conllu = parsing(text)
    df = create_df_from_conllu(conllu)
    to_drop = delete_from_df(df)
    df = df.drop(df.index[to_drop])
    return df

def start_end(df, text):
    starts = []
    ends = []
    end = 0
    for index, row in df.iterrows():
        form = row['Form']
        #print(form)
        start = end + text.find(form)
        end = start + len(form)
        text = text[text.find(form)+len(form):]
        #print(form, start, end)
        starts.append(start)
        ends.append(end)
    return starts, ends


def add_ann(ann, df, t, grid):
    strings_in_file = []
    tx = ann
    for index, row in df.iterrows():
        s1 = 'T' + str(t) + '\t' + str(row['UPosTag']) + ' ' + str(row['START']) + ' ' + str(row['END']) + '\t' + str(row['Form']) + '\n'
        s2 = '#'+ str(grid) + '\tAnnotatorNotes T' + str(t) + "\tlemma = '" + str(row['Lemma']) + "'\n"
        tx += s1 + s2
        t += 1
        grid += 1
    return tx

def itog(path_txt):
    file_name = re.search('(.+/.+?).txt', path_txt).group(1)
    ann = opening(file_name + '.ann')
    text = opening(path_txt)
    if detect(text) == 'en':
        df = df_changing(text)
        t, grid = open_ann_find_end(ann)
        starts, ends = start_end(df, text)
        df['START'] = starts
        df['END'] = ends
        tx = add_ann(ann, df, t, grid)
        with open(file_name + '.ann', 'w') as f:
            f.write(tx)
    else:
        print(path_txt)