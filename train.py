# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import time
from functools import partial
import json

import numpy as np
import paddle
from model import SentenceTransformer

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup
from paddlenlp.datasets import MapDataset


# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# fmt: on


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)



def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query = example["query"]

    query_encoded_inputs = tokenizer(text=query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_token_type_ids = query_encoded_inputs["token_type_ids"]

    # 假设 example 中包含了 feature 数据
    feature = example["feature"]  # 从数据中获取 feature

    # 需要确保 feature 是一个 1x1024 的 numpy 数组
    feature = np.array(feature, dtype="float32").reshape((1, 1024))

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return query_input_ids, query_token_type_ids, feature, label
    else:
        return query_input_ids, query_token_type_ids, feature

def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

def encode_api(api, model, tokenizer, max_seq_length=512):
    inputs = tokenizer(api, return_tensors="pd", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    # Get embedding
    with paddle.no_grad():
        api_embedding = model(query_input_ids=input_ids, query_token_type_ids=token_type_ids, isencode=True)

    return api_embedding

def accuracy(logits, labels):
    # 转为numpy
    logits = logits.numpy()
    labels = labels.numpy()

    # 按行从大排序索引
    logits_index = np.argsort(-logits, axis=0)
    # 取前五个
    logits_index = logits_index[:5, :]
    # 如果一行中labels为1，那么就是正确的
    correct = 0
    # 统计正确的个数
    for i in range(logits_index.shape[1]):
        if labels[logits_index[:, i], i].any():
            correct += 1

    total = logits_index.shape[0]*logits_index.shape[1]
    return correct/total
        


    




def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    # 按行读取数据
    with open('./all.tsv', "r", encoding="utf8") as f:
        lines = f.readlines()

    example=[]
    api_list = []
    with open('./all.json', 'r') as f:
        embeddings = json.load(f)
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("\t")
        if len(lines[i]) != 4:
            continue
        if lines[i][3] in embeddings:
            if lines[i][1] not in api_list:
                api_list.append(lines[i][1])
            feature = embeddings[lines[i][3]]
            lines[i] = {"query": lines[i][0],'api':lines[i][1], "feature":feature}
            example.append(lines[i])

    labels=-np.ones([len(api_list), len(example)])
    for i in range(len(example)):
        labels[api_list.index(example[i]['api'])][i]=1

    # 去除api
    for i in range(len(example)):
        example[i].pop('api')

    # 加入label
    for i in range(len(example)):
        example[i]['label']=labels[:,i]


    # 划分训练集和验证集
    train_ds = MapDataset(example[:int(len(example) * 0.9)])
    dev_ds = MapDataset(example[int(len(example) * 0.9):])



    pretrained_model = AutoModel.from_pretrained("bert-base-uncased")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Stack(dtype="float32"),  # feature
        Stack()
    ): [data for data in fn(samples)]
    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    dev_data_loader = create_dataloader(
        dev_ds, mode="dev", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    model = SentenceTransformer(pretrained_model)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    criterion = paddle.nn.loss.MSELoss()

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            title_embeddings=[]
            for api in api_list:
                embedding = encode_api(api, model, tokenizer, args.max_seq_length)
                title_embeddings.append(embedding)
            title_embeddings = paddle.concat(title_embeddings, axis=0)
            query_input_ids, query_token_type_ids, feature, labels= batch
            logits = model(
                query_input_ids=query_input_ids,
                query_token_type_ids=query_token_type_ids,
                feature=feature,
                title_embeddings=title_embeddings
            )
            labels = paddle.to_tensor(labels)
            labels = paddle.squeeze(labels, axis=[1])
            labels = paddle.to_tensor(labels, dtype='float32')
            loss = criterion(logits, labels)
            acc = accuracy(logits, labels)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train))
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % 100 == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
