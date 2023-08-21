# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Encoder training on single GPU"""
import argparse
import unittest
from functools import partial

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.io import DataLoader
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer

import transformer_engine.paddle as te
from transformer_engine.paddle.fp8 import is_fp8_available


class Net(nn.Layer):
    """NLP Encoder"""
    num_embed: int

    def __init__(self, num_embed, use_te=False):
        super().__init__()
        self.num_embed = num_embed
        self.use_te = use_te
        self.hidden_size = 1024
        self.num_heads = 16
        self.intermediate_size = 4096
        self.num_layers = 12

        self.embedding = nn.Embedding(num_embeddings=self.num_embed, embedding_dim=self.hidden_size)
        backend = 'transformer_engine'
        if self.use_te is False:
            backend = "paddle"


#            self.encoder_layer = nn.TransformerEncoderLayer(
#                self.hidden_size,
#                self.num_heads,
#                self.intermediate_size,
#                dropout=0,
#                activation='gelu',
#                attn_dropout=0,
#                act_dropout=0,
#                normalize_before=False)

        self.encoder = nn.LayerList([
            te.TransformerLayer(
                self.hidden_size,
                self.intermediate_size,
                self.num_heads,
                layernorm_epsilon=1e-5,
                hidden_dropout=0.1,
                attention_dropout=0.1,
                self_attn_mask_type='padding',
                apply_residual_connection_post_layernorm=False,
                output_layernorm=True,
                layer_type='encoder',
                backend=backend)
            for _ in range(self.num_layers)
        ])


        self.linear1 = nn.Linear(self.hidden_size * 512, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = paddle.cast(x, dtype='bfloat16')
        for layer in self.encoder:
            x = layer(x, mask)
        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def train(args, model, train_data_loader, optimizer, epoch, use_fp8):
    """Training function."""
    model.train()
    for batch_id, (data, mask, labels) in enumerate(train_data_loader):
        with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
            with te.fp8_autocast(enabled=use_fp8):
                outputs = model(data, mask)
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()

        if batch_id % args.log_interval == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_id * len(data)}/{len(train_data_loader.dataset)} "
                  f"({100. * batch_id / len(train_data_loader):.0f}%)]\t"
                  f"Loss: {loss.item():.6f}")
            if args.dry_run:
                return loss.item()
    return loss.item()


def evaluate(model, test_loader, epoch, use_fp8):
    """Testing function."""
    model.eval()
    metric = Accuracy()
    metric.reset()

    with paddle.no_grad():
        for data, mask, labels in test_loader:
            with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
                with te.fp8_autocast(enabled=use_fp8):
                    outputs = model(data, mask)
                acc = metric.compute(outputs, labels)
            metric.update(acc)
    print(f"Epoch[{epoch}] - accuracy: {metric.accumulate():.6f}")
    return metric.accumulate()


def calibrate(model, test_loader):
    """Calibration function."""
    model.eval()

    with paddle.no_grad():
        for data, _ in test_loader:
            with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
                with te.fp8_autocast(enabled=False, calibrating=True):
                    _ = model(data)


def convert_example(example, tokenizer, max_length=128):
    """convert example"""
    labels = np.array([example["labels"]], dtype="int64")
    example = tokenizer(example["sentence"], padding='max_length', max_length=max_length, return_token_type_ids=False)
    mask = np.zeros((1, max_length, max_length), dtype='bool')
    input_ids = example["input_ids"]
    return {
        "input_ids": input_ids,
        "mask": mask,
        "labels": labels,
    }


def load_data(batch_size, max_seqlen, tokenizer):
    """create dataloader"""
    print("Loading data")
    train_ds = load_dataset("glue", "cola", splits="train")
    validation_ds = load_dataset("glue", "cola", splits="dev")

    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=max_seqlen)
    train_ds = train_ds.map(trans_func, lazy=True)
    validation_ds = validation_ds.map(trans_func, lazy=True)

    train_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    validation_sampler = paddle.io.BatchSampler(validation_ds, batch_size=batch_size, shuffle=False)

    return train_ds, validation_ds, train_sampler, validation_sampler


def encoder_parser(args):
    """Training settings."""
    parser = argparse.ArgumentParser(description="Paddle Encoder Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        metavar="N",
        help="maximum sequence length (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--use-te",
                        action="store_true",
                        default=False,
                        help="Use Transformer Engine")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-fp8",
                        action="store_true",
                        default=False,
                        help="Use FP8 for inference and training without recalibration")

    return parser.parse_args(args)


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    # train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "mask": Pad(axis=0),
        "labels": Stack(dtype="int64"),
    }): fn(samples)
    train_ds, dev_ds, train_sampler, dev_sampler = load_data(args.batch_size, args.max_seq_len,
                                                             tokenizer)

    train_data_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=batchify_fn,
    )
    dev_data_loader = paddle.io.DataLoader(
        dev_ds,
        batch_sampler=dev_sampler,
        collate_fn=batchify_fn,
    )

    model = Net(30528, args.use_te)
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    model = paddle.amp.decorate(models=model, level='O2', dtype='bfloat16')

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, train_data_loader, optimizer, epoch, args.use_fp8)
        acc = evaluate(model, dev_data_loader, epoch, args.use_fp8)

#    if args.use_fp8_infer and not args.use_fp8:
#        calibrate(model, dev_data_loader)
#
#    if args.save_model or args.use_fp8_infer:
#        paddle.save(model.state_dict(), "mnist_cnn.pdparams")
#        print('Eval with reloaded checkpoint : fp8=' + str(args.use_fp8))
#        weights = paddle.load("mnist_cnn.pdparams")
#        model.set_state_dict(weights)
#        acc = evaluate(model, dev_data_loader, 0, args.use_fp8)

    return loss, acc


class TestEncoder(unittest.TestCase):
    """Encoder unittests"""

    gpu_has_fp8, reason = is_fp8_available()

    @classmethod
    def setUpClass(cls):
        """Run 4 epochs for testing"""
        cls.args = encoder_parser(["--epochs", "3"])

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        actual = train_and_evaluate(self.args)
        #assert actual[0] < 0.45 and actual[1] > 0.79

    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        self.args.use_fp8 = True
        actual = train_and_evaluate(self.args)
        #assert actual[0] < 0.45 and actual[1] > 0.79


if __name__ == "__main__":
    paddle.fluid.core.nvprof_enable_record_event()
    train_and_evaluate(encoder_parser(None))
