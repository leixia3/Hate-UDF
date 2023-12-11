import math
import os
import re
import shutil

import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
import pickle as pkl

from typing import Tuple


def generate_split_col(total_num) -> list:
    if total_num % 18 != 0:
        raise ValueError('total_num must be a multiple of 18')
    # train : val : test = 16 : 1 : 1
    rate = (16, 1, 1)
    train_num = rate[0] * total_num // sum(rate)
    dev_num = rate[1] * total_num // sum(rate)
    test_num = rate[2] * total_num // sum(rate)

    split = (['train' for i in range(train_num)]
            + ['dev_seen' for i in range(dev_num)]
            + ['test_seen' for i in range(test_num)])

    return split


def load_pkl(path: Path):
    data = pkl.load(open(path, 'rb'))
    return data


def load_pro_cap_infos(questions: list, pro_cap_path: Path):
    result_files = {
        q: {
            **load_pkl(pro_cap_path.joinpath('train_' + q + '.pkl')),
            **load_pkl(pro_cap_path.joinpath('test_' + q + '.pkl'))
        }
        for q in questions
    }
    valid = ['valid_person', 'valid_animal']
    for v in valid:
        result_files[v] = {
            **load_pkl(pro_cap_path.joinpath('train_' + v + '.pkl')),
            **load_pkl(pro_cap_path.joinpath('test_' + v + '.pkl'))
        }
    return result_files
    # print(result_files[list(result_files.keys())[0]])


def caption_to_pro_cap(img, caption, entity, race, pro_cap_infos: Tuple[list, dict]):
    ADD_ENTITY = True
    ADD_RACE = True

    questions, result_files = pro_cap_infos
    ext = []
    person_flag = True
    animal_flag = True
    person = result_files['valid_person'][img].lower()
    if person.startswith('no'):
        person_flag = False
    animal = result_files['valid_animal'][img].lower()
    if animal.startswith('no'):
        animal_flag = False
    for q in questions:
        if person_flag == False and q in ['race', 'gender',
                                          'country', 'valid_disable']:
            continue
        if animal_flag == False and q == 'animal':
            continue
        if q in ['valid_person', 'valid_animal']:
            continue
        info = result_files[q][img]
        if q == 'valid_disable':
            if info.startswith('no'):
                continue
            else:
                ext.append('there is a disabled person')
        else:
            ext.append(info)
    ext = ' [SEP] '.join(ext)
    caption += ' [SEP] ' + ext
    if ADD_ENTITY:
        caption += ' [SEP] ' + entity + ' [SEP] '
    if ADD_RACE:
        caption += ' [SEP] ' + race + ' [SEP] '

    return caption.strip()


def load_raw_data(src: Path, sort_rule: str):
    train_data_path = src.joinpath('train.json')
    test_data_path = src.joinpath('test.json')
    train_data = pd.read_json(train_data_path)
    test_data = pd.read_json(test_data_path)
    combined_data = pd.concat([train_data, test_data])
    combined_data.reset_index(inplace=True)
    # 计算要删除的行数
    n = len(combined_data) % 18
    # 随机选择n行数据
    drop_indices = combined_data.sample(n).index
    # 删除这些行
    combined_data = combined_data.drop(drop_indices)
    combined_data.sort_values(by='img', key=(lambda x: [int(re.search(sort_rule, item).group(1)) for item in x]), inplace=True)
    combined_data.reset_index(inplace=True)
    return combined_data


def get_target_img(src: Path, sort_rule: str):
    raw_data = load_raw_data(src, sort_rule)
    images = raw_data['img'].tolist()
    return images


def move_img(src: Path, dest: Path, sort_rule):
    for image in get_target_img(src, sort_rule):
        shutil.copy(src.joinpath('images', image), dest.joinpath('img', image))


def get_similarity(text1, text2, nlp, method='spacy'):
    """
    src: https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python
    """

    if method == 'jaccard':
        text1 = set(text1.lower().split(" "))
        text2 = set(text2.lower().split(" "))
        score = len(text1.intersection(text2)) / len(text1.union(text2))
    elif method == 'spacy':
        embed1 = nlp(text1)
        embed2 = nlp(text2)
        score = embed1.similarity(embed2)
    else:
        raise ValueError

    return score


def generate_pseudo_text_map(row_data: pd.DataFrame):
    import spacy
    nlp = spacy.load("en_core_web_md")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('../distilbert-base-nli-mean-tokens')

    meme_idx_to_text = row_data['text'].to_dict()
    meme_idxs = list(meme_idx_to_text.keys())
    scores = np.zeros((len(meme_idxs), len(meme_idxs)))

    for i in tqdm(range(len(meme_idxs))):
        for j in range(i + 1, len(meme_idxs)):
            text_i = meme_idx_to_text[meme_idxs[i]]
            text_j = meme_idx_to_text[meme_idxs[j]]
            score = get_similarity(text_i, text_j, nlp, method='jaccard')
            scores[i, j] = score
            scores[j, i] = score

    sentences = list(meme_idx_to_text.values())
    sentence_embeddings = model.encode(sentences)
    print(sentence_embeddings.shape, type(sentence_embeddings))

    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
    scores_tr = cosine_similarity(sentence_embeddings)
    for i in range(len(scores_tr)):
        scores_tr[i, i] = 0
    print(scores_tr.shape)
    print(scores_tr.min(), scores_tr.max())

    i_idxs, j_idxs = np.where((scores > 0.7) & (scores_tr > 0.7))
    print(len(i_idxs))
    for i, j in zip(i_idxs[:5], j_idxs[:5]):
        print(meme_idxs[i], meme_idxs[j], f'jaccard: {scores[i, j]}; tr: {scores_tr[i, j]}')
        print(meme_idx_to_text[meme_idxs[i]], '\n', meme_idx_to_text[meme_idxs[j]], '\n')

    meme_idxs = list(meme_idx_to_text.keys())
    pseudo_text_idx_to_meme_idxs = {}
    for i, j in zip(i_idxs, j_idxs):
        for k in range(len(pseudo_text_idx_to_meme_idxs)):
            if meme_idxs[i] in pseudo_text_idx_to_meme_idxs[k] or meme_idxs[j] in pseudo_text_idx_to_meme_idxs[k]:
                pseudo_text_idx_to_meme_idxs[k].update({meme_idxs[i], meme_idxs[j]})
                break
        else:
            pseudo_text_idx_to_meme_idxs[len(pseudo_text_idx_to_meme_idxs)] = {meme_idxs[i], meme_idxs[j]}
    print(len(pseudo_text_idx_to_meme_idxs))

    meme_idx_to_psuedo_text_idx = {}
    for pseudo_text_idx, meme_idxs in pseudo_text_idx_to_meme_idxs.items():
        for meme_idx in meme_idxs:
            meme_idx_to_psuedo_text_idx[meme_idx] = pseudo_text_idx
    print(len(meme_idx_to_psuedo_text_idx))

    # for meme idxs that are not covered
    non_covered_meme_idxs = set(row_data['id'].values) - set(meme_idx_to_psuedo_text_idx.keys())
    for i, non_covered_meme_idx in enumerate(non_covered_meme_idxs, start=len(pseudo_text_idx_to_meme_idxs)):
        meme_idx_to_psuedo_text_idx[non_covered_meme_idx] = i
    print(len(meme_idx_to_psuedo_text_idx))
    return meme_idx_to_psuedo_text_idx


def generate_pseudo_img_map(src: Path, row_data: pd.DataFrame):
    import torch
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    x = torch.randn([1, 3, 224, 224])
    output = feature_extractor(x)
    print(output.shape)

    from PIL import Image
    import torchvision.transforms as T

    transforms = T.Compose(
        [T.Resize(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    imgs = []
    for img_fp in tqdm(row_data['img'].values.tolist()):
        img = Image.open(src.joinpath('images', img_fp)).convert('RGB').resize((224, 224))
        img = transforms(img).unsqueeze(dim=0)
        imgs.append(img)

    imgs = torch.cat(imgs, dim=0)
    print(imgs.shape)

    batch_size = 18
    assert len(imgs) % batch_size == 0
    num_batches = len(imgs) // batch_size
    imgs_features = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        imgs_features_batch = feature_extractor(imgs[start_idx:end_idx])
        imgs_features_batch = imgs_features_batch.squeeze().detach().numpy()
        imgs_features.append(imgs_features_batch)
    imgs_features = np.concatenate(imgs_features)
    print(imgs_features.shape)

    from sklearn.metrics.pairwise import cosine_similarity
    scores_img = cosine_similarity(imgs_features)
    for i in range(len(scores_img)):
        scores_img[i, i] = 0
    print(scores_img.shape)
    print(scores_img.min(), scores_img.max())

    i_idxs, j_idxs = np.where((scores_img > 0.92) & (scores_img < 1))
    print(len(i_idxs))
    meme_idx_to_img = row_data['img'].to_dict()

    meme_idxs = list(meme_idx_to_img.keys())
    pseudo_img_idx_to_meme_idxs = {}
    for i, j in zip(i_idxs, j_idxs):
        for k in range(len(pseudo_img_idx_to_meme_idxs)):
            if meme_idxs[i] in pseudo_img_idx_to_meme_idxs[k] or meme_idxs[j] in pseudo_img_idx_to_meme_idxs[k]:
                pseudo_img_idx_to_meme_idxs[k].update({meme_idxs[i], meme_idxs[j]})
                break
        else:
            pseudo_img_idx_to_meme_idxs[len(pseudo_img_idx_to_meme_idxs)] = {meme_idxs[i], meme_idxs[j]}
    print(len(pseudo_img_idx_to_meme_idxs))

    meme_idx_to_pseudo_img_idx = {}
    for pseudo_img_idx, meme_idxs in pseudo_img_idx_to_meme_idxs.items():
        for meme_idx in meme_idxs:
            meme_idx_to_pseudo_img_idx[meme_idx] = pseudo_img_idx
    print(len(meme_idx_to_pseudo_img_idx))

    # for meme idxs that are not covered
    non_covered_meme_idxs = set(row_data['id'].values) - set(meme_idx_to_pseudo_img_idx.keys())
    for i, non_covered_meme_idx in enumerate(non_covered_meme_idxs, start=len(pseudo_img_idx_to_meme_idxs)):
        meme_idx_to_pseudo_img_idx[non_covered_meme_idx] = i
    print(len(meme_idx_to_pseudo_img_idx))

    return meme_idx_to_pseudo_img_idx


# 生成新的csv文件为数据集
def generate_csv(src: Path, csv_path: Path, sort_rule: str, cap_path: Path, pro_cap_path: Path, questions: list):
    """
    id img label text psudo_text_idx psudo_img_idx caption pro_cap=(caption+' [SEP] ' + pro_cap_infos) split
    """
    result = pd.DataFrame()
    img_to_caption = {
        **load_pkl(cap_path.joinpath('train_generic.pkl')),
        **load_pkl(cap_path.joinpath('test_generic.pkl'))
    }
    raw_data = load_raw_data(src, sort_rule)
    pro_cap_infos = load_pro_cap_infos(questions, pro_cap_path)

    idx_to_text = {idx: text for idx, text in enumerate(raw_data['clean_sent'].unique())}
    text_to_idx = {v: k for k, v in idx_to_text.items()}
    result['id'] = raw_data['img'].apply(lambda x: re.search(sort_rule, x).group(1))
    result['img'] = 'img/' + raw_data['img']
    result['label'] = raw_data['label']
    result['text'] = raw_data['clean_sent']
    result['text_idx'] = result['text'].map(text_to_idx)
    result['caption'] = result['img'].map(img_to_caption)

    result['entity'] = raw_data['entity']
    result['race'] = raw_data['race']

    result.index = result['id']
    result.index.name = None

    result['pro_cap'] = result.apply(lambda row: caption_to_pro_cap(row['img'], row['caption'], row['entity'], row['race'], (questions, pro_cap_infos)), axis=1)
    result['split'] = generate_split_col(len(result))
    result.drop(['entity', 'race'], axis=1, inplace=True)

    meme_idx_to_pseudo_text_idx = generate_pseudo_text_map(result)
    meme_idx_to_pseudo_img_idx = generate_pseudo_img_map(src, result)
    result['pseudo_text_idx'] = result['id'].map(meme_idx_to_pseudo_text_idx)
    result['pseudo_img_idx'] = result['id'].map(meme_idx_to_pseudo_img_idx)

    result = result.reset_index()

    result.to_csv(csv_path, index=False)


if __name__ == '__main__':
    ProCapRoot = Path(r'D:\Program\Python\ML\Ask-Captions')
    dataset2path = {
        'harm': {
            'src': Path('../harm/data'),
            'dest': Path(r'D:\Program\Python\hateclipper-main\data\Harm_covide19'),
            'sort_rule': r'\w+_\w+_(\d+).png'
        },
        'mimc': {
            'src': Path('../mimc/data'),
            'dest': Path(r'D:\Program\Python\hateclipper-main\data\misogynous_meme_detection'),
            'sort_rule': r'(\d+).jpg'
        }
    }

    for DATASET in ('harm', 'mimc'):
        DATASET_FILE_PATH = dataset2path[DATASET]['dest'].joinpath('info.csv')
        CAPTION_PATH = ProCapRoot.joinpath('Captions', DATASET)
        PRO_CAP_PATH = ProCapRoot.joinpath('Longer-Longer-Captions', DATASET)
        QUESTIONS = ['race', 'gender', 'country', 'animal', 'valid_disable', 'religion']

        generate_csv(dataset2path[DATASET]['src'], DATASET_FILE_PATH, dataset2path[DATASET]['sort_rule'], CAPTION_PATH, PRO_CAP_PATH, QUESTIONS)
        # move_img(dataset2path[DATASET]['src'], dataset2path[DATASET]['dest'], dataset2path[DATASET]['sort_rule'])
