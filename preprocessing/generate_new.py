import json
import os
import pickle as pkl
from pathlib import Path

import pandas as pd
ROOT = Path('../')
differences_path = ROOT.joinpath('data/hateful_memes/preprocessing/differences.csv')
new_info_file_path = ROOT.joinpath('data/hateful_memes/hateful_memes_expanded.csv')
old_info_file_path_base = ROOT.joinpath('data/hateful_memes/ask_captions/mem')
result_path = ROOT.joinpath('data/hateful_memes/')
questions = ['race', 'gender', 'country', 'animal', 'valid_disable', 'religion']
questions_with_valid = questions + ['valid_person', 'valid_animal']
cols = 'disability_gold_pc,nationality_gold_pc,pc_empty_gold_pc,race_gold_pc,religion_gold_pc,sex_gold_pc,attack_empty_gold_attack,contempt_gold_attack,dehumanizing_gold_attack,exclusion_gold_attack,inciting_violence_gold_attack,inferiority_gold_attack,mocking_gold_attack,slurs_gold_attack,disability_pc,nationality_pc,pc_empty_pc,race_pc,religion_pc,sex_pc,attack_empty_attack,contempt_attack,dehumanizing_attack,exclusion_attack,inciting_violence_attack,inferiority_attack,mocking_attack,slurs_attack'


def read_json(path):
    assert Path.exists(path), 'Does not exist : {}'.format(path)
    data = json.load(open(path, 'rb'))
    '''in anet-qa returns a list'''
    return data


def load_pkl(path):
    data = pkl.load(open(path, 'rb'))
    return data


def get_old_pic_names(question):
    train_path = old_info_file_path_base.joinpath('train_' + question + '.pkl')
    test_path = old_info_file_path_base.joinpath('test_' + question + '.pkl')
    train_pics = list(load_pkl(train_path).keys())
    test_pics = list(load_pkl(test_path).keys())
    set1 = set(train_pics)
    set2 = set(test_pics)
    if not set1.intersection(set2):
        pics = train_pics + test_pics
        pics.sort()
        return pics
    else:
        raise ValueError()


def get_each_question_pics():
    pics = []
    for question in questions_with_valid:
        pics.append(get_old_pic_names(question))

    if all(row == pics[0] for row in pics):
        return pics[0]
    else:
        raise ValueError('Exist different pics in each question!')


def get_new_pic_names():
    df = pd.read_csv(new_info_file_path)
    pics = df['img'].apply(lambda x: x.split('/')[1]).tolist()
    set1 = set(pics)
    # print(f'{len(pics)=}, {len(set1)=}')
    pics = list(set1)
    pics.sort()

    return pics, df


def find_out():
    new, row = get_new_pic_names()
    old = get_each_question_pics()
    isHit = []
    split = []
    for pic in new:
        isHit.append(pic in old)
        split.append(row[row['img'].apply(lambda x: x.split('/')[1]) == pic]['split'].values[0])
    data = {
        'split': split,
        'new': new,
        'isHit': isHit
    }
    df = pd.DataFrame(data)

    # Write DataFrame to CSV
    df.to_csv(differences_path, index=False, sep=',')

# check if the new dataset is the same as the old one
# find_out()


def load_entries(questions_):
    ADD_ENTITY = True
    ADD_RACE = True

    path = old_info_file_path_base.joinpath('mem.json')
    data = read_json(path)

    cap_path = new_info_file_path

    result_files = {
        q: {
            **load_pkl(old_info_file_path_base.joinpath('train_' + q + '.pkl')),
            **load_pkl(old_info_file_path_base.joinpath('test_' + q + '.pkl'))
        }
        for q in questions_
    }
    # result_files = {q: load_pkl(os.path.join(
    #     '../../Ask-Captions/' + self.opt.LONG + 'Captions',
    #     self.opt.DATASET,
    #     mode + '_' + q + '.pkl'))
    #     for q in questions_}

    valid = ['valid_person', 'valid_animal']
    for v in valid:
        result_files[v] = {
            **load_pkl(old_info_file_path_base.joinpath('train_' + v + '.pkl')),
            **load_pkl(old_info_file_path_base.joinpath('test_' + v + '.pkl'))
        }
    print(len(result_files))
    for key in result_files.keys():
        print(key, len(result_files[key]))
        # result_files[v] = load_pkl(os.path.join(
        #     '../../Ask-Captions/' + self.opt.LONG + 'Captions',
        #     self.opt.DATASET,
        #     mode + '_' + v + '.pkl'))

    # captions = load_pkl(cap_path)
    captions_map = pd.read_csv(cap_path, usecols=['img', 'caption'])
    captions_map['img'] = captions_map['img'].apply(lambda x: x.split('/')[1])

    result = {}
    for k, row in enumerate(data):
        # image的文件名
        img = row['img']
        cap = captions_map[captions_map['img'] == img]['caption'].values[0]
        ext = []
        person_flag = True
        animal_flag = True
        person = result_files['valid_person'][row['img']].lower()
        if person.startswith('no'):
            person_flag = False
        animal = result_files['valid_animal'][row['img']].lower()
        if animal.startswith('no'):
            animal_flag = False
        for q in questions_:
            if person_flag == False and q in ['race', 'gender',
                                              'country', 'valid_disable']:
                continue
            if animal_flag == False and q == 'animal':
                continue
            if q in ['valid_person', 'valid_animal']:
                continue
            info = result_files[q][row['img']]
            if q == 'valid_disable':
                if info.startswith('no'):
                    continue
                else:
                    ext.append('there is a disabled person')
            else:
                ext.append(info)
        ext = ' [SEP] '.join(ext)
        cap = cap + ' [SEP] ' + ext

        # whether using external knowledge
        if ADD_ENTITY:
            cap = cap + ' [SEP] ' + row['entity'] + ' [SEP] '
        if ADD_RACE:
            cap = cap + ' [SEP] ' + row['race'] + ' [SEP] '

        result[img] = cap.strip()
        # entry = {
        #     'cap': cap.strip(),  # generic_cap, ask_cap, meme_aware_cap, external knowledge
        #     'img': img
        # }
        # entries.append(entry)
    return result
    # return entries


def generate_new_dataset(cols, with_out_val):
    cols = cols.split(',')
    df_info = pd.read_csv(differences_path)
    df = pd.read_csv(new_info_file_path)
    for col in cols:
        df[col] = df[col].astype('Int64')
    # Delete anonymous column
    df.drop(df.columns[0], axis=1, inplace=True)
    print('Finish Delete anonymous column.')
    # delete row split column
    df.drop('split', axis=1, inplace=True)
    print('Finish delete row split column.')
    # delete miss column
    isHit = [df_info[pic == df_info['new']]['isHit'].bool() for pic in df['img'].apply(lambda x: x.split('/')[1]).tolist()]
    df['isHit'] = isHit
    df.drop(df[~df['isHit']].index, inplace=True)
    df.drop('isHit', axis=1, inplace=True)
    print('Finish delete miss column.')
    # add Pro-Cap
    data = load_entries(questions)
    df['pro_cap'] = [data[pic] for pic in df['img'].apply(lambda x: x.split('/')[1]).tolist()]
    print('Finish add Pro-Cap.')
    # add new split column
    if with_out_val:
        split = (['train' for i in range(8500)]
                 + ['test_seen' for i in range(500)]
        )
    else:
              # train,  dev_seen, text_seen, new_total
        rate = (8500,       500,      1000,      9000)
        split = (['train' for i in range(int(rate[0]/sum(rate[:-1])*rate[-1]))]
                 + ['dev_seen' for i in range(int(rate[1]/sum(rate[:-1])*rate[-1]))]
                 + ['test_seen' for i in range(int(rate[2]/sum(rate[:-1])*rate[-1]))])

    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    df_shuffled['split'] = split
    print('Finish add new split column.')
    # write to CSV
    df_shuffled.to_csv(result_path.joinpath('modified_hateful_memes_expanded.csv'), sep=',')


generate_new_dataset(cols, True)

