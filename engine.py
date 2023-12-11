from datetime import datetime
import json
from pyexpat import features
import copy
import math
from sys import prefix
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from torch import Tensor

from transformers import CLIPModel, AutoConfig, AutoModel


class CLIPClassifier(pl.LightningModule):
    # 方法
    def weight_training(self, batch, image_features, text_features, pro_cap_features):
        # 变更单模态预测方式
        # output = {
        #     'image': dict(),
        #     'text': dict(),
        #     'pro_cap': dict()
        # }
        #
        # image_features_pre_output = self.pre_output_image(image_features)
        # image_logits = self.output_image(image_features_pre_output).squeeze(dim=1)  # [batch_size, 1]
        # pred_image_weights = torch.sigmoid(image_logits)
        # preds_image = (pred_image_weights >= 0.5).long()# shape = (batch_size)
        # pred_image_weights = pred_image_weights.unsqueeze(1)  # shape = (batch_size, 1)
        #
        # output['image']['loss'] = self.cross_entropy_loss(image_logits, batch['labels'].float())
        # output['image']['accuracy'] = self.acc(preds_image, batch['labels'])
        # output['image']['auroc'] = self.auroc(pred_image_weights, batch['labels'])
        #
        # text_features_pre_output = self.pre_output_text(text_features)
        # text_logits = self.output_text(text_features_pre_output).squeeze(dim=1)  # [batch_size, 1]
        # pred_text_weights = torch.sigmoid(text_logits)  # shape = (batch_size)
        # preds_text = (pred_text_weights >= 0.5).long()
        # pred_text_weights = pred_text_weights.unsqueeze(1)  # shape = (batch_size, 1)
        #
        # output['text']['loss'] = self.cross_entropy_loss(text_logits, batch['labels'].float())
        # output['text']['accuracy'] = self.acc(preds_text, batch['labels'])
        # output['text']['auroc'] = self.auroc(pred_text_weights, batch['labels'])
        #
        # if pro_cap_features is not None:
        #     pro_cap_features_pre_output = self.pre_output_pro_cap(pro_cap_features)
        #     pro_cap_logits = self.output_pro_cap(pro_cap_features_pre_output).squeeze(dim=1)
        #     pred_pro_cap_weights = torch.sigmoid(pro_cap_logits)
        #     preds_pro_cap = (pred_pro_cap_weights >= 0.5).long()
        #     pred_pro_cap_weights = pred_pro_cap_weights.unsqueeze(1) # shape = (batch_size, 1)
        #
        #     output['pro_cap']['loss'] = self.cross_entropy_loss(pro_cap_logits, batch['labels'].float())
        #     output['pro_cap']['accuracy'] = self.acc(preds_pro_cap, batch['labels'])
        #     output['pro_cap']['auroc'] = self.auroc(pred_pro_cap_weights, batch['labels'])
        #
        #     # 归一化
        #     # 首先拼接
        #     tmp_tensor = torch.cat((pred_image_weights, pred_text_weights, pred_pro_cap_weights), 1)
        #     # 然后调用F.softmax函数，按列（一次三个值）做标准化，使得一个batch_size下image_feature_w, text_feature_w和pro_cap_feature_w权重和=1
        #     softmax_tmp_tensor = F.softmax(tmp_tensor, dim=1)
        #     # 最后分离新tensor并依次赋值给image_feature_w，text_feature_w和pro_cap_feature_w
        #     tensor_chunked = torch.chunk(softmax_tmp_tensor.t(), chunks=3, dim=0)
        #     image_feature_w = tensor_chunked[0].t()
        #     text_feature_w = tensor_chunked[1].t()
        #     pro_cap_feature_w = tensor_chunked[2].t()
        #
        #     output['image']['weight'] = image_feature_w
        #     output['text']['weight'] = text_feature_w
        #     output['pro_cap']['weight'] = pro_cap_feature_w
        # else:
        #     # 归一化
        #     # 首先拼接
        #     tmp_tensor = torch.cat((pred_image_weights, pred_text_weights), 1)
        #     # 然后调用F.softmax函数，按列（一次三个值）做标准化，使得一个batch_size下image_feature_w和text_feature_w权重和=1
        #     softmax_tmp_tensor = F.softmax(tmp_tensor, dim=1)
        #     # 最后分离新tensor并依次赋值给image_feature_w和text_feature_w
        #     tensor_chunked = torch.chunk(softmax_tmp_tensor.t(), chunks=2, dim=0)
        #     image_feature_w = tensor_chunked[0].t()
        #     text_feature_w = tensor_chunked[1].t()
        #
        #     output['image']['weight'] = image_feature_w
        #     output['text']['weight'] = text_feature_w
        #
        # return output
        # 输出
        output = {}

        for feature_type, features, pre_output_layer, output_layer in zip(
                ['image', 'text', 'pro_cap'],
                [image_features, text_features, pro_cap_features],
                [self.pre_output_image, self.pre_output_text, self.pre_output_pro_cap],
                [self.output_image, self.output_text, self.output_pro_cap]
        ):
            if features is not None:
                features_pre_output = pre_output_layer(features)
                logits = output_layer(features_pre_output).squeeze(dim=1)
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                preds_proxy = preds_proxy.unsqueeze(1)

                if self.weight_generator == 'acc':
                    weight = self.acc(preds, batch['labels']).repeat(len(batch['idx_memes']), 1)
                elif self.weight_generator == 'direct':
                    weight = preds_proxy
                elif self.weight_generator.endswith('fixed'):
                    # torch.where(condition, True Value Choice, False Value Choice)
                    # 当满足condition时，选择True Value Choice，否则选择False Value Choice
                    # 注意这个操作是element-wise的
                    pre_weight = torch.where(preds_proxy >= 0.5, preds_proxy, 1 - preds_proxy)
                    if self.weight_generator == 'fixed':
                        weight = pre_weight
                    elif self.weight_generator == 'l_after_fixed':
                        # pre_weight过一层线性层
                        weight = self.weight_linear(pre_weight)
                    else:
                        raise ValueError()
                else:
                    raise ValueError()
                # # torch.where(condition, True Value Choice, False Value Choice)
                # # 当满足condition时，选择True Value Choice，否则选择False Value Choice
                # # 注意这个操作是element-wise的
                # pre_weight = torch.where(preds_proxy >= 0.5, preds_proxy, 1 - preds_proxy)
                # # pre_weight过一层线性层
                #
                # weight = self.weight_linear(pre_weight)
                output[feature_type] = {
                    'loss': self.cross_entropy_loss(logits, batch['labels'].float()),
                    'accuracy': self.acc(preds, batch['labels']),
                    'auroc': self.auroc(preds_proxy, batch['labels']),
                    'pure_weight': preds_proxy,
                    # 'pre_weight': pre_weight,
                    # 返回一个不会梯度下降的新张量
                    'weight': weight.detach(),
                    # 'weight': self.acc(preds, batch['labels']).repeat(len(batch['idx_memes']), 1)
                }
        # 归一化
        # if 'pro_cap' in output:
        #     tmp_tensor = torch.cat(
        #         (output['image']['weight'], output['text']['weight'], output['pro_cap']['weight']), 1)
        # else:
        #     tmp_tensor = torch.cat((output['image']['weight'], output['text']['weight']), 1)
        #
        # softmax_tmp_tensor = F.softmax(tmp_tensor, dim=1)
        # tensor_chunked = torch.chunk(softmax_tmp_tensor.t(), chunks=len(output), dim=0)
        #
        # for feature_type, tensor in zip(output, tensor_chunked):
        #     output[feature_type]['weight'] = tensor.t()

        # TODO: TEST BEGIN
        # with open('log_weight.json', 'at') as file:
        #     file.write(', ' + json.dumps({
        #         **{
        #             f'{feature_type}_{estimator}': output[feature_type][estimator].tolist() for estimator in
        #             ['loss', 'accuracy', 'auroc'] for feature_type in
        #             (['image', 'text', 'pro_cap'] if 'pro_cap' in output else ['image', 'text'])
        #         },
        #         'batch': [{
        #             'img': batch['idx_memes'].tolist()[idx],
        #             **{**{
        #                     f"{feature_type}_weight": output[feature_type]['weight'].tolist()[idx][0] for feature_type in
        #                     (['image', 'text', 'pro_cap'] if 'pro_cap' in output else ['image', 'text'])
        #                 },
        #                **{
        #                    f"{feature_type}_pure_weight": output[feature_type]['pure_weight'].tolist()[idx][0] for
        #                    feature_type in (['image', 'text', 'pro_cap'] if 'pro_cap' in output else ['image', 'text'])
        #                },
        #             }
        #         } for idx in range(len(batch['idx_memes']))],
        #     }))
        #
        # print(f'{output["image"]["weight"].shape=}, {output["image"]["accuracy"]=}')
        with open('./log_weights_without_pro_cap.jsonl', 'at') as file:
            file.write('\n'.join([json.dumps(item) for item in [{
                feature_type: output[feature_type]['weight'].tolist()[i] for feature_type in output.keys()
            } for i in range(16)]]) + '\n')
        # TODO: TEST END
        return output

    def __init__(self, args, fine_grained_labels, compute_fine_grained_metrics):
        super().__init__()
        # self.args = args
        self.caption_mode = args.caption_mode
        self.use_pretrained_map = args.use_pretrained_map
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.fusion = args.fusion
        self.debug = args.debug
        self.with_pro_cap = args.with_pro_cap
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # self.weight_image_loss = args.weight_image_loss
        # self.weight_text_loss = args.weight_text_loss
        self.weight_fine_grained_loss = args.weight_fine_grained_loss
        self.weight_super_loss = args.weight_super_loss
        self.fine_grained_labels = fine_grained_labels
        self.compute_fine_grained_metrics = compute_fine_grained_metrics

        # for tamil dataset
        self.text_encoder_name = args.text_encoder
        self.dataset = args.dataset

        # self.acc = torchmetrics.Accuracy()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        if self.dataset == 'prop':
            self.auroc = torchmetrics.AUROC(num_classes=22)
            self.precision_score = torchmetrics.Precision(mdmc_average='global')
            self.recall = torchmetrics.Recall(mdmc_average='global')
            self.f1 = torchmetrics.F1Score(mdmc_average='global')
        else:
            # self.auroc = torchmetrics.AUROC()
            self.auroc = torchmetrics.AUROC(task="binary")
            # self.precision_score = torchmetrics.Precision()
            self.precision_score = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
            # self.recall = torchmetrics.Recall()
            self.recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=3)
            # self.f1 = torchmetrics.F1Score()
            self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=3)
        pretrained_path = '/root/autodl-tmp/clip-vit-large-patch14'
        self.clip = CLIPModel.from_pretrained(pretrained_path)
        # self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)
        if args.local_pretrained_weights != 'none':
            state_dict = torch.load(args.local_pretrained_weights)['state_dict']
            state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('clip')}
            self.clip.load_state_dict(state_dict)
        if args.image_encoder == 'clip':
            self.image_encoder = copy.deepcopy(self.clip.vision_model)
        else:
            raise ValueError()
        if args.text_encoder == 'clip':
            # TODO: 看看encoder具体是做什么的
            self.text_encoder = copy.deepcopy(self.clip.text_model)
        elif args.text_encoder:
            config = AutoConfig.from_pretrained(args.text_encoder, output_hidden_states=True)
            self.text_encoder = AutoModel.from_pretrained(args.text_encoder, config=config)
        else:
            raise ValueError()

        if self.use_pretrained_map:
            self.image_map = nn.Sequential(
                copy.deepcopy(self.clip.visual_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
            self.text_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
            self.pro_cap_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
        else:
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim),
                                nn.Dropout(p=args.drop_probs[0])]
            text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim),
                               nn.Dropout(p=args.drop_probs[0])]
            # 引入Pro-Cap映射层
            pro_cap_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim),
                                  nn.Dropout(p=args.drop_probs[0])]
            for _ in range(1, self.num_mapping_layers):
                image_map_layers.extend(
                    [nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])
                text_map_layers.extend(
                    [nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])
                pro_cap_map_layers.extend(
                    [nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=args.drop_probs[0])])

            self.image_map = nn.Sequential(*image_map_layers)
            self.text_map = nn.Sequential(*text_map_layers)
            self.pro_cap_map = nn.Sequential(*pro_cap_map_layers)

        self.weight_linear = nn.Linear(1, 1)

        fusion = args.fusion
        # TODO: 目前设计weighted模式下pre_output_input_dim与非weighted下一致，请确认是否正确
        if fusion.startswith('weighted'):
            self.weight_generator = args.weight_generator
            fusion = fusion.replace('weighted_', '')
        if fusion == 'sum':
            pre_output_input_dim = self.map_dim
        elif fusion in ['align', 'align_shuffle']:
            pre_output_input_dim = self.map_dim
        elif fusion == 'concat':
            if self.with_pro_cap:
                pre_output_input_dim = self.map_dim * 3
            else:
                pre_output_input_dim = self.map_dim * 2
        elif fusion.startswith('cross'):
            pre_output_input_dim = self.map_dim ** 2
        elif fusion == 'align_concat':
            pre_output_input_dim = self.map_dim * (4 if self.with_pro_cap else 3)
        elif fusion == 'attention_m':
            self.gen_query = nn.Linear(self.map_dim, self.map_dim // 4)
            self.gen_key = nn.Linear(self.map_dim, self.map_dim // 4)
            self.soft = nn.Softmax(dim=1)
            pre_output_input_dim = self.map_dim * 2
        else:  # 看看args.fusion是什么
            print("args.fusion is ", fusion)
        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1:  # first pre-output layer
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers):  # next pre-output layers
            pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            self.output = nn.Linear(output_input_dim, 1)
        elif self.dataset == 'prop':
            self.output = nn.Linear(output_input_dim, 22)

        image_pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        text_pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        pro_cap_pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        for _ in range(self.num_pre_output_layers):  # next pre-output layers
            image_pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            text_pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            pro_cap_pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
        self.pre_output_image = nn.Sequential(*image_pre_output_layers)
        self.pre_output_text = nn.Sequential(*text_pre_output_layers)
        self.pre_output_pro_cap = nn.Sequential(*pro_cap_pre_output_layers)
        if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            self.output_image = nn.Linear(output_input_dim, 1)
            self.output_text = nn.Linear(output_input_dim, 1)
            self.output_pro_cap = nn.Linear(output_input_dim, 1)
        elif self.dataset == 'prop':
            self.output_image = nn.Linear(output_input_dim, 22)
            self.output_text = nn.Linear(output_input_dim, 22)
            self.output_pro_cap = nn.Linear(output_input_dim, 1)

        # if self.weight_image_loss > 0:
        #     pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        #     for _ in range(self.num_pre_output_layers): # next pre-output layers
        #         pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
        #     self.pre_output_image = nn.Sequential(*pre_output_layers)
        #     if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
        #         self.output_image = nn.Linear(output_input_dim, 1)
        #     elif self.dataset == 'prop':
        #         self.output_image = nn.Linear(output_input_dim, 22)
        # if self.weight_text_loss > 0:
        #     pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        #     for _ in range(self.num_pre_output_layers): # next pre-output layers
        #         pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
        #     self.pre_output_text = nn.Sequential(*pre_output_layers)
        #     if self.dataset in ['original', 'masked', 'inpainted', 'tamil']:
        #         self.output_text = nn.Linear(output_input_dim, 1)
        #     elif self.dataset == 'prop':
        #         self.output_text = nn.Linear(output_input_dim, 22)

        if self.fine_grained_labels:
            if self.dataset in ['original', 'masked', 'inpainted']:
                self.output_pc1 = nn.Linear(output_input_dim, 1)
                self.output_pc2 = nn.Linear(output_input_dim, 1)
                self.output_pc3 = nn.Linear(output_input_dim, 1)
                self.output_pc4 = nn.Linear(output_input_dim, 1)
                self.output_pc5 = nn.Linear(output_input_dim, 1)
                self.output_pc6 = nn.Linear(output_input_dim, 1)
                self.output_attack1 = nn.Linear(output_input_dim, 1)
                self.output_attack2 = nn.Linear(output_input_dim, 1)
                self.output_attack3 = nn.Linear(output_input_dim, 1)
                self.output_attack4 = nn.Linear(output_input_dim, 1)
                self.output_attack5 = nn.Linear(output_input_dim, 1)
                self.output_attack6 = nn.Linear(output_input_dim, 1)
                self.output_attack7 = nn.Linear(output_input_dim, 1)
                self.output_attack8 = nn.Linear(output_input_dim, 1)
                self.outputs_fine_grained = [self.output_pc1, self.output_pc2, self.output_pc3, self.output_pc4,
                                             self.output_pc5, self.output_pc6,
                                             self.output_attack1, self.output_attack2, self.output_attack3,
                                             self.output_attack4, self.output_attack5, self.output_attack6,
                                             self.output_attack7, self.output_attack8]
                self.output_super = nn.Linear(15, 1)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip
        if self.caption_mode == 'replace_image':
            del self.image_encoder, self.image_map

    def forward(self, batch):
        if self.caption_mode != "replace_image":
            image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
            image_features = self.image_map(image_features)
        elif self.text_encoder_name == 'clip':
            image_features = self.text_encoder(input_ids=batch['input_ids_caption'],
                                               attention_mask=batch['attention_mask_caption']).pooler_output
            image_features = self.text_map(image_features)
        else:
            text_feats = \
                self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[
                    "hidden_states"][
                    -1][:, 0, :]
            image_features = self.text_map(image_features)

        if self.text_encoder_name == 'clip':
            text_features = self.text_encoder(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask']).pooler_output
        else:
            text_features = \
                self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[
                    "hidden_states"][
                    -1][:, 0, :]
        text_features = self.text_map(text_features)

        if self.caption_mode.startswith('parallel'):
            caption_features = self.text_encoder(input_ids=batch['input_ids_caption'],
                                                 attention_mask=batch['attention_mask_caption']).pooler_output
            caption_features = self.text_map(caption_features)

        image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1)  # [batch_size, d]
        if self.caption_mode.startswith('parallel'):
            caption_features = F.normalize(caption_features, p=2, dim=1)

        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)  # [batch_size, 2*d]
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [batch_size, d, d]
            if self.fusion == 'cross_nd':
                mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
                del mask
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        elif self.fusion == 'align_concat':
            features = torch.cat([torch.mul(image_features, text_features), image_features, text_features],
                                 dim=1)  # [batch_size, 3*d]
        elif self.fusion == 'attention_m':
            q1 = F.relu(self.gen_query(image_features))
            k1 = F.relu(self.gen_key(image_features))
            q2 = F.relu(self.gen_query(text_features))
            k2 = F.relu(self.gen_key(text_features))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float())  # prob
            prob_1 = wt_i1_i2[:, 0]
            prob_2 = wt_i1_i2[:, 1]
            wtd_i1 = image_features * prob_1[:, None]
            wtd_i2 = text_features * prob_2[:, None]
            features = torch.cat((wtd_i1, wtd_i2), 1)  # [batch_size, 2*d]
        else:
            raise ValueError()

        if self.caption_mode.startswith('parallel'):
            if self.fusion in ['align', 'align_shuffle']:
                features_parallel = torch.mul(caption_features, text_features)  # [batch_size, d]
            elif self.fusion == 'concat':
                features_parallel = torch.cat([caption_features, text_features], dim=1)  # [batch_size, 2*d]
            elif self.fusion.startswith('cross'):
                features_parallel = torch.bmm(caption_features.unsqueeze(2),
                                              text_features.unsqueeze(1))  # [batch_size, d, d]
                if self.fusion == 'cross_nd':
                    mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                    features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
                    del mask
                features_parallel = features_parallel.reshape(features_parallel.shape[0], -1)  # [batch_size, d*d]
            elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features],
                                     dim=1)  # [batch_size, 3*d]
            elif self.fusion == 'attention_m':
                q1 = F.relu(self.gen_query(image_features))
                k1 = F.relu(self.gen_key(image_features))
                q2 = F.relu(self.gen_query(text_features))
                k2 = F.relu(self.gen_key(text_features))
                score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
                score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
                wt_score1_score2_mat = torch.cat((score1, score2), 1)
                wt_i1_i2 = self.soft(wt_score1_score2_mat.float())  # prob
                prob_1 = wt_i1_i2[:, 0]
                prob_2 = wt_i1_i2[:, 1]
                wtd_i1 = image_features * prob_1[:, None]
                wtd_i2 = text_features * prob_2[:, None]
                features = torch.cat((wtd_i1, wtd_i2), 1)  # [batch_size, 2*d]
            else:
                raise ValueError()

            if self.caption_mode == 'parallel_max':
                features = torch.maximum(features, features_parallel)
            elif self.caption_mode == 'parallel_mean':
                features = (features + features_parallel) / 2.0
            elif self.caption_mode == 'parallel_align':
                features = torch.mul(features, features_parallel)
            else:
                raise ValueError()

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch, batch_idx, calling_function='validation'):
        #  resize()

        if self.caption_mode != "replace_image":
            image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
            image_features = self.image_map(image_features)

        else:
            image_features = self.text_encoder(input_ids=batch['input_ids_caption'],
                                               attention_mask=batch['attention_mask_caption']).pooler_output
            image_features = self.text_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        # 引入Pro-Cap特征值
        if self.with_pro_cap:
            pro_cap_features = self.text_encoder(input_ids=batch['input_ids_pro_cap'],
                                                 attention_mask=batch['attention_mask_pro_cap']).pooler_output
            pro_cap_features = self.pro_cap_map(pro_cap_features)

        if self.caption_mode.startswith('parallel'):
            caption_features = self.text_encoder(input_ids=batch['input_ids_caption'],
                                                 attention_mask=batch['attention_mask_caption']).pooler_output
            caption_features = self.text_map(caption_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        if self.with_pro_cap:
            pro_cap_features = F.normalize(pro_cap_features, p=2, dim=1)
        if self.caption_mode.startswith('parallel'):
            caption_features = F.normalize(caption_features, p=2, dim=1)
        output = {}

        # if self.weight_image_loss > 0:
        #     features_pre_output = self.pre_output_image(image_features)
        #     logits = self.output_image(features_pre_output).squeeze(dim=1) # [batch_size, 1]
        #     preds_proxy = torch.sigmoid(logits)
        #     preds = (preds_proxy >= 0.5).long()
        #
        #     output['image_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        #     output['image_accuracy'] = self.acc(preds, batch['labels'])
        #     output['image_auroc'] = self.auroc(preds_proxy, batch['labels'])
        #
        # if self.weight_text_loss > 0:
        #     features_pre_output = self.pre_output_text(text_features)
        #     logits = self.output_text(features_pre_output).squeeze(dim=1) # [batch_size, 1]
        #     preds_proxy = torch.sigmoid(logits)
        #     preds = (preds_proxy >= 0.5).long()
        #
        #     output['text_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        #     output['text_accuracy'] = self.acc(preds, batch['labels'])
        #     output['text_auroc'] = self.auroc(preds_proxy, batch['labels'])

        # Pro-Cap在特征向量级别融合
        if self.fusion == 'sum':
            if self.with_pro_cap:
                features = image_features + text_features + pro_cap_features
            else:
                features = image_features + text_features
        elif self.fusion in ['align', 'align_shuffle']:
            if self.with_pro_cap:
                features = torch.mul(torch.mul(image_features, text_features), pro_cap_features)
            else:
                features = torch.mul(image_features, text_features)
        elif self.fusion == 'concat':
            if self.with_pro_cap:
                features = torch.cat([image_features, text_features, pro_cap_features], dim=1)
            else:
                features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [16, d, d]
            if self.fusion == 'cross_nd':
                mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
                del mask
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        elif self.fusion == 'align_concat':
            if self.with_pro_cap:
                features = torch.cat([torch.mul(torch.mul(image_features, text_features), pro_cap_features), image_features, text_features, pro_cap_features], dim=1)
            else:
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features],
                                     dim=1)  # [batch_size, 3*d]
        elif self.fusion.startswith('weighted'):
            # Simplified code
            weights_and_estimate = self.weight_training(batch, image_features, text_features,
                                                        pro_cap_features if self.with_pro_cap else None)
            weighted_features = ([features * weights_and_estimate[feature_type]['weight'] for feature_type, features in
                                  zip(['image', 'text'], [image_features, text_features])]
                                 + [pro_cap_features * weights_and_estimate['pro_cap'][
                        'weight'] if self.with_pro_cap else None])
            weighted_features = [feature for feature in weighted_features if feature is not None]
            # 标准化后的特征向量，len(normalized_features) = 2 or 3，取决于有无Pro-Cap
            normalized_features = [F.normalize(feature, p=2, dim=1) for feature in weighted_features if
                                   feature is not None]
            if self.fusion == 'weighted_sum':
                features = sum(normalized_features)
            elif self.fusion in ['weighted_align', 'weighted_align_shuffle']:
                features = torch.prod(torch.stack(normalized_features), dim=0)
            elif self.fusion == 'weighted_align_concat':
                features = torch.cat([torch.prod(torch.stack(normalized_features), dim=0), *normalized_features], dim=1)
            elif self.fusion == 'weighted_concat':
                features = torch.cat(normalized_features, dim=1)
            # # 这里只融合了两个向量
            # elif self.fusion.startswith('weighted_cross'):
            #     features = torch.bmm(normalized_features[0].unsqueeze(2), normalized_features[1].unsqueeze(1))
            #     if self.fusion == 'weighted_cross_nd':
            #         mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
            #         features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
            #     features = features.reshape(features.shape[0], -1)
            else:
                raise ValueError()

            # weights_and_estimate = self.weight_taining(batch, image_features, text_features,
            #                                            pro_cap_features if self.with_pro_cap else None)
            #
            # weighted_features = ([features * weight
            #                       for features, weight in zip([image_features, text_features],
            #                                                   [
            #                                                       weights_and_estimate[feature_type]['weight']
            #                                                       for feature_type in ['image', 'text']
            #                                                   ])
            #                       ]
            #                      + [pro_cap_features * weights_and_estimate['pro_cap'][
            #             'weight'] if self.with_pro_cap else None])
            #
            # normalized_features = [F.normalize(feature, p=2, dim=1) for feature in weighted_features if
            #                        feature is not None]

            # # TODO: TEST BEGIN
            # if self.debug != 'none':
            #     image_feature_w = image_feature_w.tolist()
            #     text_feature_w = text_feature_w.tolist()
            #     print(f'{self.debug=}\n{image_feature_w=}\n{text_feature_w=}')
            #     with open('log_pointed_meme_weights.json', 'at') as file:
            #         file.write(json.dumps({
            #             'img': self.debug,
            #             'image_feature_w': image_feature_w,
            #             'text_feature_w': text_feature_w
            #         }))
            #     exit()
            # # TODO: TEST END

            # if self.fusion in ['weighted_align', 'weighted_align_shuffle']:
            #     # [batch_size, d]
            #     if self.with_pro_cap:
            #         features = torch.mul(torch.mul(weighted_image_features, weighted_text_features),
            #                              weighted_pro_cap_features)
            #     else:
            #         features = torch.mul(weighted_image_features, weighted_text_features)
            # elif self.fusion == 'weighted_concat':
            #     # [batch_size, 2*d]
            #     if self.with_pro_cap:
            #         features = torch.cat([weighted_image_features, weighted_text_features, weighted_pro_cap_features],
            #                              dim=1)
            #     else:
            #         features = torch.cat([weighted_image_features, weighted_text_features], dim=1)
            # # 引入三模态Cross融合机制
            # elif self.fusion.startswith('weighted_cross'):
            #     features = torch.bmm(weighted_image_features.unsqueeze(2),
            #                          weighted_text_features.unsqueeze(1))  # [16, d, d]
            #     if self.fusion == 'weighted_cross_nd':
            #         mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
            #         features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
            #         del mask
            #     features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
            # else:
            #     raise ValueError()



        elif self.fusion == 'attention_m':
            q1 = F.relu(self.gen_query(image_features))
            k1 = F.relu(self.gen_key(image_features))
            q2 = F.relu(self.gen_query(text_features))
            k2 = F.relu(self.gen_key(text_features))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float())  # prob
            prob_1 = wt_i1_i2[:, 0]
            prob_2 = wt_i1_i2[:, 1]
            wtd_i1 = image_features * prob_1[:, None]
            wtd_i2 = text_features * prob_2[:, None]
            features = torch.cat((wtd_i1, wtd_i2), 1)  # [batch_size, 2*d]
        else:
            raise ValueError()

        if self.caption_mode.startswith('parallel'):
            if self.fusion in ['align', 'align_shuffle']:
                features_parallel = torch.mul(caption_features, text_features)  # [batch_size, d]
            elif self.fusion == 'concat':
                features_parallel = torch.cat([caption_features, text_features], dim=1)  # [batch_size, 2*d]
            elif self.fusion.startswith('cross'):
                features_parallel = torch.bmm(caption_features.unsqueeze(2),
                                              text_features.unsqueeze(1))  # [batch_size, d, d]
                if self.fusion == 'cross_nd':
                    mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                    features[mask] = torch.zeros(features.shape[0] * self.map_dim, device=features.device)
                    del mask
                features_parallel = features_parallel.reshape(features_parallel.shape[0], -1)  # [batch_size, d*d]
            elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features],
                                     dim=1)  # [batch_size, 3*d]
            elif self.fusion == 'attention_m':
                q1 = F.relu(self.gen_query(image_features))
                k1 = F.relu(self.gen_key(image_features))
                q2 = F.relu(self.gen_query(text_features))
                k2 = F.relu(self.gen_key(text_features))
                score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
                score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
                wt_score1_score2_mat = torch.cat((score1, score2), 1)
                wt_i1_i2 = self.soft(wt_score1_score2_mat.float())  # prob
                prob_1 = wt_i1_i2[:, 0]
                prob_2 = wt_i1_i2[:, 1]
                wtd_i1 = image_features * prob_1[:, None]
                wtd_i2 = text_features * prob_2[:, None]
                features = torch.cat((wtd_i1, wtd_i2), 1)  # [batch_size, 2*d]
            else:
                raise ValueError()

            if self.caption_mode == 'parallel_max':
                features = torch.maximum(features, features_parallel)
            elif self.caption_mode == 'parallel_mean':
                features = (features + features_parallel) / 2.0
            elif self.caption_mode == 'parallel_align':
                features = torch.mul(features, features_parallel)
            else:
                raise ValueError()

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1(or)n]
        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            logits_for_super = [torch.relu(logits)]
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()
        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])
        if self.fusion.startswith('weighted'):
            output['image_loss'] = weights_and_estimate['image']['loss']
            output['image_accuracy'] = weights_and_estimate['image']['accuracy']
            output['image_auroc'] = weights_and_estimate['image']['auroc']

            output['text_loss'] = weights_and_estimate['text']['loss']
            output['text_accuracy'] = weights_and_estimate['text']['accuracy']
            output['text_auroc'] = weights_and_estimate['text']['auroc']
            if self.with_pro_cap:
                output['pro_cap_loss'] = weights_and_estimate['pro_cap']['loss']
                output['pro_cap_accuracy'] = weights_and_estimate['pro_cap']['accuracy']
                output['pro_cap_auroc'] = weights_and_estimate['pro_cap']['auroc']

        if self.dataset in ['tamil', 'prop']:
            output['precision'] = self.precision_score(preds, batch['labels'])
            output['recall'] = self.recall(preds, batch['labels'])
            output['f1'] = self.f1(preds, batch['labels'])

        if calling_function == 'training' and self.fine_grained_labels and self.dataset in ['original', 'masked',
                                                                                            'inpainted']:
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features_pre_output).squeeze(dim=1)
                logits_for_super.append(torch.relu(logits))
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits,
                                                                               batch[fine_grained_label].float())
            logits_for_super = torch.stack(logits_for_super, dim=1)  # [batch_size, 15]
            logits = self.output_super(logits_for_super).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()
            output['super_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output['super_accuracy'] = self.acc(preds, batch['labels'])
            output['super_auroc'] = self.auroc(preds_proxy, batch['labels'])

        elif calling_function == 'validation' and self.fine_grained_labels and self.dataset in ['original', 'masked',
                                                                                                'inpainted']:
            for fine_grained_label, output_fine_grained in zip(self.fine_grained_labels, self.outputs_fine_grained):
                logits = output_fine_grained(features_pre_output).squeeze(dim=1)
                logits_for_super.append(torch.relu(logits))
                preds_proxy = torch.sigmoid(logits)
                preds = (preds_proxy >= 0.5).long()
                output[f'{fine_grained_label}_loss'] = self.cross_entropy_loss(logits,
                                                                               batch[fine_grained_label].float())
                output[f'{fine_grained_label}_accuracy'] = self.acc(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_auroc'] = self.auroc(preds_proxy, batch[fine_grained_label])
                output[f'{fine_grained_label}_precision'] = self.precision_score(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_recall'] = self.recall(preds, batch[fine_grained_label])
                output[f'{fine_grained_label}_f1'] = self.f1(preds, batch[fine_grained_label])
            logits_for_super = torch.stack(logits_for_super, dim=1)  # [batch_size, 15]
            logits = self.output_super(logits_for_super).squeeze(dim=1)
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()
            output[f'super_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output[f'super_accuracy'] = self.acc(preds, batch['labels'])
            output[f'super_auroc'] = self.auroc(preds_proxy, batch['labels'])
            output[f'super_precision'] = self.precision_score(preds, batch['labels'])
            output[f'super_recall'] = self.recall(preds, batch['labels'])
            output[f'super_f1'] = self.f1(preds, batch['labels'])

        elif calling_function == 'visualisation-v1':
            return image_features, text_features

        elif calling_function == 'visualisation-v2':
            return features

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='training')
        # if self.weight_image_loss > 0:
        #     image_loss = output['image_loss']
        # else:
        #     image_loss = 0
        #
        # if self.weight_text_loss > 0:
        #     text_loss = output['text_loss']
        # else:
        #     text_loss = 0

        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            fine_grained_loss = 0
            for fine_grained_label in self.fine_grained_labels:
                fine_grained_loss += output[f'{fine_grained_label}_loss']
            fine_grained_loss /= len(self.fine_grained_labels)
            super_loss = output['super_loss']
        else:
            fine_grained_loss = 0.0
            super_loss = 0.0


        # 加入单模态的损失函数，忽略正则化项
        loss_items = [output[item] for item in output.keys() if item.endswith('loss')]
        total_loss = sum(loss_items) / len(loss_items)

        # total_loss = (output['loss'] + output['image_loss'] + output['text_loss'] + (output['pro_cap_loss'] if self.with_pro_cap else 0)) / (4 if self.with_pro_cap else 3)

        # total_loss = (output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss +
        #               self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss)

        # total_loss = (output['loss'] + output['image_loss'] + output['text_loss']
        #               + output['pro_cap_loss'] if self.with_pro_cap else 0
        #               + self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss)

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log('train/precision', output['precision'])
            self.log('train/recall', output['recall'])
            self.log('train/f1', output['f1'])

        if self.fusion.startswith('weighted'):
            self.log('train/image_loss', output['image_loss'])
            self.log('train/image_accuracy', output['image_accuracy'])
            self.log('train/image_auroc', output['image_auroc'])

            self.log('train/text_loss', output['text_loss'])
            self.log('train/text_accuracy', output['text_accuracy'])
            self.log('train/text_auroc', output['text_auroc'])
            if self.with_pro_cap:
                self.log('train/pro_cap_loss', output['pro_cap_loss'])
                self.log('train/pro_cap_accuracy', output['pro_cap_accuracy'])
                self.log('train/pro_cap_auroc', output['pro_cap_auroc'])

        # if self.weight_image_loss > 0:
        #     self.log('train/image_loss', image_loss)
        # if self.weight_text_loss > 0:
        #     self.log('train/text_loss', text_loss)
        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            self.log('train/fine_grained_loss', fine_grained_loss)
            self.log('train/super_loss', super_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='validation')

        # if self.weight_image_loss > 0:
        #     image_loss = output['image_loss']
        # else:
        #     image_loss = 0
        #
        # if self.weight_text_loss > 0:
        #     text_loss = output['text_loss']
        # else:
        #     text_loss = 0

        if self.fine_grained_labels and self.compute_fine_grained_metrics and self.dataset in ['original', 'masked',
                                                                                               'inpainted']:
            fine_grained_loss = torch.mean(
                torch.Tensor([output[f'{fine_grained_label}_loss'] for fine_grained_label in self.fine_grained_labels]))
            super_loss = output['super_loss']
        else:
            fine_grained_loss = 0.0
            super_loss = 0.0

            # 加入单模态的损失函数，忽略正则化项
            loss_items = [output[item] for item in output.keys() if item.endswith('loss')]
            total_loss = sum(loss_items) / len(loss_items)

            # total_loss = (output['loss'] + output['image_loss'] + output['text_loss']
            #               + output['pro_cap_loss'] if self.with_pro_cap else 0) / (4 if self.with_pro_cap else 3)

            # total_loss = (output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss +
            #               self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss)

            # total_loss = (output['loss'] + output['image_loss'] + output['text_loss']
            #               + output['pro_cap_loss'] if self.with_pro_cap else 0
            #               + self.weight_fine_grained_loss * fine_grained_loss + self.weight_super_loss * super_loss)
        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'])
        self.log(f'val/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log('val/precision', output['precision'])
            self.log('val/recall', output['recall'])
            self.log('val/f1', output['f1'])

        if self.fusion.startswith('weighted'):
            self.log('train/image_loss', output['image_loss'])
            self.log('train/image_accuracy', output['image_accuracy'])
            self.log('train/image_auroc', output['image_auroc'])

            self.log('train/text_loss', output['text_loss'])
            self.log('train/text_accuracy', output['text_accuracy'])
            self.log('train/text_auroc', output['text_auroc'])
            if self.with_pro_cap:
                self.log('train/pro_cap_loss', output['pro_cap_loss'])
                self.log('train/pro_cap_accuracy', output['pro_cap_accuracy'])
                self.log('train/pro_cap_auroc', output['pro_cap_auroc'])

        # if self.weight_image_loss > 0:
        #     self.log(f'val/image_loss', image_loss)
        # if self.weight_text_loss > 0:
        #     self.log(f'val/text_loss', text_loss)

        if self.fine_grained_labels and self.compute_fine_grained_metrics and self.dataset in ['original', 'masked',
                                                                                               'inpainted']:
            self.log(f'val/fine_grained_loss', fine_grained_loss)
            self.log(f'val/super_loss', super_loss)

            for fine_grained_label in self.fine_grained_labels:
                self.log(f'val-fine-grained/{fine_grained_label}_accuracy', output[f'{fine_grained_label}_accuracy'])
                self.log(f'val-fine-grained/{fine_grained_label}_auroc', output[f'{fine_grained_label}_auroc'])
                self.log(f'val-fine-grained/{fine_grained_label}_precision', output[f'{fine_grained_label}_precision'])
                self.log(f'val-fine-grained/{fine_grained_label}_recall', output[f'{fine_grained_label}_recall'])
                self.log(f'val-fine-grained/{fine_grained_label}_f1', output[f'{fine_grained_label}_f1'])

            self.log(f'val/super_loss', output['super_loss'])
            self.log(f'val/super_accuracy', output['super_accuracy'])
            self.log(f'val/super_auroc', output['super_auroc'])

        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        prefix_map = {
            0: 'test_seen',
            1: 'dev_seen',
            2: 'dev_unseen',
            3: 'test_unseen'
        }
        prefix = prefix_map[dataloader_idx]
        if dataloader_idx == 0:
            calling_function = 'training'
        elif dataloader_idx == 1:
            calling_function = 'validation'

        output = self.common_step(batch, batch_idx, calling_function=calling_function)

        self.log(f'{prefix}/accuracy', output['accuracy'])
        self.log(f'{prefix}/auroc', output['auroc'])
        if self.dataset in ['tamil', 'prop']:
            self.log(f'{prefix}/precision', output['precision'])
            self.log(f'{prefix}/recall', output['recall'])
            self.log(f'{prefix}/f1', output['f1'])

        if self.fine_grained_labels and self.dataset in ['original', 'masked', 'inpainted']:
            self.log(f'{prefix}/super_accuracy', output['super_accuracy'])
            self.log(f'{prefix}/super_auroc', output['super_auroc'])

            if dataloader_idx == 1:
                for fine_grained_label in self.fine_grained_labels:
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_accuracy',
                             output[f'{fine_grained_label}_accuracy'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_auroc', output[f'{fine_grained_label}_auroc'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_precision',
                             output[f'{fine_grained_label}_precision'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_recall',
                             output[f'{fine_grained_label}_recall'])
                    self.log(f'{prefix}-fine-grained/{fine_grained_label}_f1', output[f'{fine_grained_label}_f1'])

        return output

    def training_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def validation_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def test_epoch_end(self, validation_step_outputs):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args, fine_grained_labels):
    compute_fine_grained_metrics = True
    if args.from_checkpoint != 'none':
        model = CLIPClassifier.load_from_checkpoint(checkpoint_path=args.from_checkpoint,
                                                    args=args, fine_grained_labels=fine_grained_labels,
                                                    compute_fine_grained_metrics=compute_fine_grained_metrics)
    else:
        model = CLIPClassifier(args=args, fine_grained_labels=fine_grained_labels,
                               compute_fine_grained_metrics=compute_fine_grained_metrics)

    return model
