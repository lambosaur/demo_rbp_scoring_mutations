#! /usr/bin/env python
# coding:utf8

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Union, Optional, Tuple
from typing_extensions import Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import logomaker as lm
import tensorflow as tf

# tf.enable_eager_execution() # necessary for tf version < 2
tf.compat.v1.enable_eager_execution()  # necessary for tf version < 2
tf.enable_eager_execution()


class ModelPysster:

    alphabet = "ACGT"
    unknown_nt = "N"
    alphabet_int = {k: v for v, k in enumerate(alphabet)}

    def __init__(self, model: tf.keras.models.Model):
        self.model = model

    @classmethod
    def load_from_path(cls, model_path: Union[str, os.PathLike]) -> Self:
        model = tf.keras.models.load_model(model_path)
        return cls(model=model)

    @classmethod
    def _onehot_encode(cls, seq: str) -> np.ndarray:
        seq_int = np.array([cls.alphabet_int.get(nt, 99) for nt in seq])
        seq_onehot = tf.one_hot(seq_int, depth=len(cls.alphabet_int))
        seq_onehot = seq_onehot.numpy()[None, :]
        return seq_onehot

    def predict_window(self, seq: str):
        self._validate_sequence(seq)
        seq_onehot = self._onehot_encode(seq)
        scores = self.model.predict(seq_onehot)
        return scores

    def _validate_sequence(self, seq: str) -> None:
        if not isinstance(seq, str):
            raise ValueError("<seq> should be of type <str>")

        if not len(seq) == self.model.input_shape[1]:
            raise ValueError(
                f"Input sequence should be of length {self.model.input_shape[1]}"
            )

        alphabet_full = self.alphabet + self.unknown_nt
        all_nt_in_alphabet = [nt in alphabet_full for nt in seq]

        if not all(all_nt_in_alphabet):
            # get first unrecognized nt.
            unrecognized_nt = [
                i
                for i, nt_in_alphabet in enumerate(all_nt_in_alphabet)
                if not nt_in_alphabet
            ]
            raise ValueError(
                (
                    f"Unrecognized nucleotide : {seq[unrecognized_nt[0]]} "
                    f"(alphabet: {alphabet_full})"
                )
            )

class StructIG:

    pysster_colors = {"A": "#00CC00", "C": "#0000CC", "G": "#FFB300", "T": "#CC0000",}

    def __init__(self, ig_tbl: pd.DataFrame):
        if not list(ig_tbl.columns) == ["A","C","G","T"]:
            raise ValueErrror("Expects a dataframe with columns ['A','C','G','T']")

        self.ig_tbl = ig_tbl

    def plot_logo(
        self,
        start_end: Optional[Tuple[int, int]]=None,
        title: Optional[str]=None,
        ax: Optional[mpl.axes._axes.Axes]=None,
        show_plot:bool=True,
    ) -> Optional[mpl.axes._axes.Axes]:
        if ax is None:
            fig = plt.figure(figsize=(25, 3.6))
            ax = fig.add_subplot(1, 1, 1)

        if start_end is not None:
            start, end = start_end

            if not start >= 0:
                raise ValueError("<start> should be >=0")

            if not end > start:
                raise ValueError("<end> should be >= <start>")

            if end > len(self.ig_tbl):
                raise ValueError("<end> should be <= len(self.ig_tbl)")

            tbl = self.ig_tbl.iloc[start:end, :]
        else:
            tbl = self.ig_tbl

        logo = lm.Logo(df=tbl, ax=ax, color_scheme=self.pysster_colors)

        if title is not None:
            ax.set_title(title)

        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            return ax


def interpolate_seqs(
    baseline: np.ndarray, seq: np.ndarray, alphas: tf.Tensor
) -> tf.Tensor:
    delta: np.ndarray = seq - baseline
    seqs: tf.Tensor = baseline + alphas[:, tf.newaxis, tf.newaxis] * delta
    return seqs


def compute_gradients(
    seqs: np.ndarray, model: tf.keras.models.Model, target_class_idx: int
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(seqs)
        logits = model(seqs)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]

    return tape.gradient(probs, seqs)


def integral_approximation(gradients: tf.Tensor) -> tf.Tensor:
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(
    seq: np.ndarray,
    model: tf.keras.models.Model,
    target_class_idx: int,
    m_steps: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    baseline: np.ndarray = np.zeros_like(seq)
    alphas: tf.Tensor = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    gradient_batches: tf.TensorArray = tf.TensorArray(tf.float32, size=m_steps + 1)

    alpha: int
    for alpha in range(0, len(alphas), batch_size):
        from_: int = alpha
        to: int = tf.minimum(from_ + batch_size, len(alphas))

        alpha_batch: np.ndarray = alphas[from_:to]

        interpolated_batch: np.ndarray = interpolate_seqs(baseline, seq, alpha_batch)

        gradient_batch: tf.Tensor = compute_gradients(
            seqs=interpolated_batch, model=model, target_class_idx=target_class_idx
        )

        gradient_batches: tf.TensorArray = gradient_batches.scatter(
            tf.range(from_, to), gradient_batch
        )
        total_gradients: tf.Tensor = gradient_batches.stack()
        avg_gradients = integral_approximation(gradients=total_gradients)
        igs = (seq - baseline) * avg_gradients

    return igs.numpy()[0, :, :]



def integrate_gradient_model_from_nucleotide_seq(
    seq: str, model_structure: ModelPysster,  target_class_idx:int, m_steps:int=64, batch_size:int=32
) -> StructIG:

    seq_onehot: np.ndarray = model_structure._onehot_encode(seq)

    igs: np.ndarray = integrated_gradients(
        seq=seq_onehot,
        model=model_structure.model,
        target_class_idx=0,
        m_steps=64,
        batch_size=32,
    )
    ig_tbl = pd.DataFrame(igs, columns=list(model_structure.alphabet))
    struct_ig = StructIG(ig_tbl=ig_tbl)
    return struct_ig
