"""Gradio demo — the public face of ForgeML.

Two tabs, and the second one is the point.

**Playground** is the obvious half: type a prompt, get an answer.

**The Trade-off** is what makes this a portfolio piece rather than another chat
box. It shows the full experiment table and states plainly why the deployed model
is not the highest-scoring one. Anyone can fine-tune a model; being able to
explain why you shipped the second-best one is the actual skill.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr

from forgeml.inference.predictor import load_predictor
from forgeml.logging_utils import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

PREDICTOR = load_predictor(allow_stub=True)
INFO = PREDICTOR.info()

RESULTS_PATH = Path(os.getenv("FORGEML_RESULTS", "reports/selection.json"))

EXAMPLES = [
    ["Explain overfitting in machine learning.", ""],
    ["What is the difference between LoRA and full fine-tuning?", ""],
    ["Summarise the passage in one sentence.",
     "Quantization reduces the numerical precision of model weights. It shrinks "
     "memory footprint and often speeds up inference, at some cost to accuracy."],
    ["Write a haiku about gradient descent.", ""],
]


def generate(prompt: str, context: str, max_new_tokens: int, temperature: float) -> tuple[str, str]:
    """Run one prediction and return the answer plus a small stats line."""
    if not prompt.strip():
        return "", "Enter a prompt to begin."

    result = PREDICTOR.predict(
        prompt=prompt,
        context=context or None,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )
    stats = (
        f"**{result.latency_ms:.0f} ms**  ·  "
        f"{result.completion_tokens} tokens generated  ·  "
        f"{result.prompt_tokens} prompt tokens  ·  "
        f"finish: `{result.finish_reason}`"
    )
    return result.text, stats


def _model_card_markdown() -> str:
    lines = [
        f"### {INFO.name}",
        "",
        "| | |", "|---|---|",
        f"| Base model | `{INFO.base_model}` |",
        f"| Method | {INFO.method.upper()} |",
        f"| Version | {INFO.version} |",
    ]
    if INFO.parameters:
        lines.append(f"| Parameters | {INFO.parameters / 1e6:.0f}M |")
    if INFO.quantization:
        lines.append(f"| Quantization | {INFO.quantization} |")
    if INFO.quality is not None:
        lines.append(f"| Quality | {INFO.quality:.4f} |")
    if INFO.latency_p95_ms is not None:
        lines.append(f"| Latency p95 | {INFO.latency_p95_ms:.0f} ms |")

    if not INFO.loaded:
        lines += [
            "",
            "> **No model is loaded.** Responses are placeholders. Set "
            "`FORGEML_MODEL_URI` to serve the real champion.",
        ]
    return "\n".join(lines)


def _tradeoff_markdown() -> str:
    """Render the selection result, or explain how to produce one."""
    if not RESULTS_PATH.exists():
        return (
            "### The trade-off\n\n"
            "No selection report found yet. Run the pipeline and point "
            "`FORGEML_RESULTS` at the generated `selection.json` to populate "
            "this tab.\n\n"
            "```bash\nforgeml select --output reports/selection.json\n```"
        )

    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    champion = payload.get("champion") or {}
    weights = payload.get("weights", {})

    rows = [
        "| Run | Method | Quality | Latency p95 | Memory | Size | Utility |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in payload.get("ranked", []) + payload.get("rejected", []):
        rows.append(
            f"| {candidate['run_name']} | {candidate['method']} | "
            f"{candidate['quality']:.4f} | {candidate['latency_ms']:.0f} ms | "
            f"{candidate['memory_mb']:.0f} MB | {candidate['model_size_mb']:.0f} MB | "
            f"{candidate['utility']:.4f} |"
        )

    return "\n".join(
        [
            "### Why this model, and not the highest-scoring one?",
            "",
            "Every candidate below was trained on identical data and measured "
            "through one evaluation path. Selection then applied hard constraints "
            "and ranked what survived on a weighted trade-off:",
            "",
            "```",
            *[f"{k:<12} {v}" for k, v in weights.items()],
            "```",
            "",
            *rows,
            "",
            f"**Champion:** `{champion.get('run_name', 'n/a')}` "
            f"(utility {champion.get('utility', 0):.4f})",
            "",
            "The winner frequently is *not* the best on quality. That is the "
            "entire point: a model two points better and twice as slow is the "
            "wrong model for most production budgets.",
        ]
    )


def build_ui() -> Any:
    with gr.Blocks(title="ForgeML", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ForgeML\n"
            "### LLM fine-tuning, evaluation and optimization platform\n"
            "Fine-tunes a small open model with LoRA / QLoRA, tracks every run in "
            "MLflow, measures quality **and** latency **and** memory **and** cost, "
            "then picks a champion automatically."
        )

        with gr.Tabs():
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(
                            label="Instruction", lines=3,
                            placeholder="Explain overfitting in machine learning.",
                        )
                        context = gr.Textbox(
                            label="Context (optional)", lines=3,
                            placeholder="Grounding text the answer should be based on.",
                        )
                        with gr.Row():
                            max_tokens = gr.Slider(16, 512, value=192, step=16,
                                                   label="Max new tokens")
                            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1,
                                                    label="Temperature (0 = greedy)")
                        submit = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown(_model_card_markdown())

                output = gr.Textbox(label="Response", lines=10, show_copy_button=True)
                stats = gr.Markdown()

                gr.Examples(examples=EXAMPLES, inputs=[prompt, context])

                submit.click(
                    generate,
                    inputs=[prompt, context, max_tokens, temperature],
                    outputs=[output, stats],
                )
                prompt.submit(
                    generate,
                    inputs=[prompt, context, max_tokens, temperature],
                    outputs=[output, stats],
                )

            with gr.Tab("The trade-off"):
                gr.Markdown(_tradeoff_markdown())

            with gr.Tab("How it works"):
                gr.Markdown(
                    "```\n"
                    "Dataset\n"
                    "   -> validation + hash-bucket split + content fingerprint\n"
                    "   -> fine-tuning       (baseline / LoRA / QLoRA)\n"
                    "   -> MLflow            (params, metrics, artifacts, model)\n"
                    "   -> evaluation        (quality, latency, memory, cost)\n"
                    "   -> selection         (constraints, then weighted utility)\n"
                    "   -> model registry    (@champion alias)\n"
                    "   -> this API\n"
                    "```\n\n"
                    "Source: https://github.com/kkrishna-labs/forgeml"
                )

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_ui().queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
    )
