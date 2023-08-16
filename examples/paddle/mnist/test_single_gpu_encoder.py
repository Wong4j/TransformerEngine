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
import nltk
import numpy as np
import optax
from datasets import load_dataset
from paddle.metric import Accuracy
from paddle.io import DataLoader

import transformer_engine.paddle as te
import transformer_engine.paddle.fp8 as is_fp8_available



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


        self.embedding = nn.Embedding(num_embeddings=self.num_embed, embedding_dim=self.hidden_size)
        if self.use_te == False:
            self.encoder_layer = nn.TransformerEncoderLayer(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size,
                dropout=0,
                activation='gelu',
                attn_dropout=0,
                act_dropout=0,
                normalize_before=False)
        else: 
            self.encoder_layer = te.TransformerLayer(
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                layernorm_epsilon=1e-5,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                self_attn_mask_type='padding',
                apply_residual_connection_post_layernorm=False,
                output_layernorm=True,
                layer_type='encoder',
                backend='transformer_engine')

        self.linear1 = nn.Linear(self.hidden_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.encoder_layer(x, attention_mask=mask)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def train(args, model, train_loader, optimizer, epoch, use_fp8):
    """Training function."""
    model.train()
    for batch_id, (data, labels) in enumerate(train_loader):
        with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
            with te.fp8_autocast(enabled=use_fp8):
                outputs = model(data)
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()

        if batch_id % args.log_interval == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_id * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_id / len(train_loader):.0f}%)]\t"
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
        for data, labels in test_loader:
            with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
                with te.fp8_autocast(enabled=use_fp8):
                    outputs = model(data)
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


def data_preprocess(dataset, vocab, word_id, max_seq_len):
    """Convert tokens to numbers."""
    nltk.download('punkt')
    dataset_size = len(dataset['sentence'])
    output = np.zeros((dataset_size, max_seq_len), dtype=np.int32)
    mask_3d = np.ones((dataset_size, max_seq_len, max_seq_len), dtype=np.uint8)

    for j, sentence in enumerate(dataset['sentence']):
        tokens = nltk.word_tokenize(sentence)
        tensor = output[j]

        for i, word in enumerate(tokens):
            if i >= max_seq_len:
                break

            if word not in vocab:
                vocab[word] = word_id
                tensor[i] = word_id
                word_id = word_id + 1
            else:
                tensor[i] = vocab[word]

        seq_len = min(len(tokens), max_seq_len)
        mask_2d = mask_3d[j]
        mask_2d[:seq_len, :seq_len] = 0

    new_dataset = {
        'sentence': output,
        'label': dataset['label'].astype(np.float32),
        'mask': mask_3d.reshape((dataset_size, 1, max_seq_len, max_seq_len))
    }
    return new_dataset, vocab, word_id


def get_datasets(max_seq_len):
    """Load GLUE train and test datasets into memory."""
    vocab = {}
    word_id = 0

    train_ds = load_dataset('glue', 'cola', split='train')
    train_ds.set_format(type='np')
    train_ds, vocab, word_id = data_preprocess(train_ds, vocab, word_id, max_seq_len)

    test_ds = load_dataset('glue', 'cola', split='validation')
    test_ds.set_format(type='np')
    test_ds, vocab, word_id = data_preprocess(test_ds, vocab, word_id, max_seq_len)
    return train_ds, test_ds, word_id





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
        default=32,
        metavar="N",
        help="maximum sequence length (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
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
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument("--use-fp8",
                        action="store_true",
                        default=False,
                        help="Use FP8 for inference and training without recalibration")

    return parser.parse_args(args)


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)

    input_shape = [args.batch_size, args.max_seq_len]
    mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
    label_shape = [args.batch_size]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    model = Net(num_embed, args.use_te)
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    model = paddle.amp.decorate(models=model, amp_level='O2', dtype='bfloat16')

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, train_loader, optimizer, epoch, args.use_fp8)
        acc = evaluate(model, test_loader, epoch, args.use_fp8)

    if args.use_fp8_infer and not args.use_fp8:
        calibrate(model, test_loader)

    if args.save_model or args.use_fp8_infer:
        paddle.save(model.state_dict(), "mnist_cnn.pdparams")
        print('Eval with reloaded checkpoint : fp8=' + str(args.use_fp8))
        weights = paddle.load("mnist_cnn.pdparams")
        model.set_state_dict(weights)
        acc = evaluate(model, test_loader, 0, args.use_fp8)


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
    train_and_evaluate(encoder_parser(None))
