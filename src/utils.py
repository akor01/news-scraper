import json
from colorama import Fore, Style

def load_json_file(filepath, default=None):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print_colored(f"Error loading {filepath}: {e}", Fore.RED)
        return default if default is not None else []

def save_json_file(filepath, data):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print_colored(f"Saved {len(data)} items to {filepath}", Fore.GREEN)
    except Exception as e:
        print_colored(f"Error saving to {filepath}: {e}", Fore.RED)

def print_colored(msg, color):
    print(color + str(msg) + Style.RESET_ALL)

def print_summary_table(summary_table, headers):
    try:
        from tabulate import tabulate
        print(tabulate(summary_table, headers=headers))
    except ImportError:
        # Fallback to plain print
        print(' | '.join(headers))
        for row in summary_table:
            print(' | '.join(str(x) for x in row)) 