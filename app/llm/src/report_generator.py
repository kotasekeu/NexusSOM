import os
import json
from datetime import datetime

from llm.src.context_builder import build_context
from llm.src.llm_client import OllamaClient


REPORT_PROMPT = """Based on the dataset description and SOM analysis results provided above, generate a complete analysis report.

Structure your report as follows:

## Dataset Summary
Brief overview of what this dataset contains and how many samples were analyzed.

## Map Quality
Assessment of the SOM map quality based on MQE, topographic error, and dead neurons.

## Key Findings
The most important patterns discovered — what groups exist in the data, how they differ, and what this means in domain terms.

## Cluster Analysis
For each major cluster (non-trivial size), describe:
- What samples it contains (using domain terminology)
- Key characteristics (dimension averages in domain context)
- How it differs from other clusters

## Anomalies
Notable outlier samples — what makes them unusual and what this might mean.

## Conclusions
Summary of actionable insights from this analysis.

Use specific numbers, sample IDs, and neuron references. Do not invent any data."""


def generate_report(dataset_path, model="llama3.1:8b", base_url="http://localhost:11434",
                    stream=True):
    """Generate a full analysis report from SOM results."""
    client = OllamaClient(base_url=base_url, model=model)

    if not client.is_available():
        raise ConnectionError(
            f"Ollama is not running at {base_url}. Start it with: ollama serve"
        )

    available = client.list_models()
    if not any(model in m for m in available):
        raise ValueError(
            f"Model '{model}' not found. Available: {available}. "
            f"Pull it with: ollama pull {model}"
        )

    # Build context
    system_prompt, user_context = build_context(dataset_path)
    full_prompt = f"{user_context}\n\n---\n\n{REPORT_PROMPT}"

    print(f"Context size: {len(user_context)} chars")
    print(f"Generating report with {model}...\n")

    # Generate
    if stream:
        response = client.generate(
            prompt=full_prompt,
            system=system_prompt,
            stream_callback=lambda t: print(t, end="", flush=True)
        )
        print()  # newline after streaming
    else:
        response = client.generate(prompt=full_prompt, system=system_prompt)

    # Save outputs
    _save_outputs(dataset_path, response, system_prompt, full_prompt, model)

    return response


def start_chat(dataset_path, model="llama3.1:8b", base_url="http://localhost:11434"):
    """Start an interactive chat session about the dataset."""
    client = OllamaClient(base_url=base_url, model=model)

    if not client.is_available():
        raise ConnectionError(
            f"Ollama is not running at {base_url}. Start it with: ollama serve"
        )

    system_prompt, user_context = build_context(dataset_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the dataset and SOM analysis context:\n\n{user_context}"},
        {"role": "assistant", "content": "I have the dataset and SOM analysis context. I'm ready to answer questions about this data. What would you like to know?"}
    ]

    print("Chat mode — ask questions about the dataset. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not question or question.lower() in ("exit", "quit", "q"):
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": question})

        print("Assistant: ", end="", flush=True)
        response = client.chat(
            messages=messages,
            stream_callback=lambda t: print(t, end="", flush=True)
        )
        print()

        messages.append({"role": "assistant", "content": response})


def _save_outputs(dataset_path, report, system_prompt, full_prompt, model):
    """Save report and prompt log."""
    from llm.src.context_builder import find_som_results

    som_dir = find_som_results(dataset_path)
    llm_dir = os.path.join(som_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)

    # Save report
    report_path = os.path.join(llm_dir, "report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save prompt log for traceability
    log_path = os.path.join(llm_dir, "prompt_log.json")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": full_prompt,
        "response_length": len(report)
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Prompt log saved to: {log_path}")
