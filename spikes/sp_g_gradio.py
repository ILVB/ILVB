"""SP-G: minimal Gradio app — launches locally, accepts an upload, returns an image."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np


def invert(img: np.ndarray | None) -> np.ndarray | None:
    return None if img is None else 255 - img


def build() -> gr.Blocks:
    with gr.Blocks(title="SP-G") as demo:
        inp = gr.Image(type="numpy", label="upload")
        out = gr.Image(type="numpy", label="result")
        btn = gr.Button("Run")
        btn.click(invert, inputs=inp, outputs=out, api_name="invert", concurrency_limit=1)
    return demo


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7861
    print("gradio", gr.__version__)
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=False
    )
