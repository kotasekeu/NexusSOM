import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='NexusSOM LLM Analysis — The Voice')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the SOM results directory or dataset directory')
    parser.add_argument('-m', '--mode', choices=['report', 'chat', 'pdf'], default='report',
                        help='Mode: "report" generates full analysis, "chat" interactive Q&A, "pdf" rebuilds PDF from existing report.md')
    parser.add_argument('--model', default='llama3.1:8b',
                        help='Ollama model name (default: llama3.1:8b)')
    parser.add_argument('--url', default='http://localhost:11434',
                        help='Ollama server URL (default: http://localhost:11434)')
    args = parser.parse_args()

    from llm.src.report_generator import generate_report, start_chat, build_pdf_report

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
    elif args.mode == 'pdf':
        pdf_path = build_pdf_report(args.input)
        if pdf_path:
            print(f"PDF saved to: {pdf_path}")
        else:
            print("Error: report.md not found. Run report mode first.", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
