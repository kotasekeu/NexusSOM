import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='NexusSOM LLM Analysis — The Voice')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the dataset directory (containing results/SOM/ and dataset_context.txt)')
    parser.add_argument('-m', '--mode', choices=['report', 'chat'], default='report',
                        help='Mode: "report" generates a full analysis, "chat" starts interactive Q&A')
    parser.add_argument('--model', default='llama3.1:8b',
                        help='Ollama model name (default: llama3.1:8b)')
    parser.add_argument('--url', default='http://localhost:11434',
                        help='Ollama server URL (default: http://localhost:11434)')
    args = parser.parse_args()

    from llm.src.report_generator import generate_report, start_chat

    if args.mode == 'report':
        generate_report(
            dataset_path=args.input,
            model=args.model,
            base_url=args.url
        )
    elif args.mode == 'chat':
        start_chat(
            dataset_path=args.input,
            model=args.model,
            base_url=args.url
        )


if __name__ == "__main__":
    sys.exit(main())
