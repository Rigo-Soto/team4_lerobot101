#!/usr/bin/env python3
from pathlib import Path
import argparse

OLD = "lerobot.src.lerobot"
NEW = "lerobot"


def is_import_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def fix_file(path: Path, dry_run: bool = False, backup: bool = True) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    changed = False
    new_lines = []

    for line in lines:
        # Solo cambia líneas de imports
        if is_import_line(line) and OLD in line:
            line = line.replace(OLD, NEW)
            changed = True

        new_lines.append(line)

    if changed and not dry_run:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            backup_path.write_text(text, encoding="utf-8")

        path.write_text("".join(new_lines), encoding="utf-8")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Reemplaza imports de lerobot.src.lerobot por lerobot en archivos .py."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directorio raíz donde buscar archivos .py. Default: directorio actual.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra qué archivos cambiaría, sin modificarlos.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="No crea archivos .bak antes de modificar.",
    )

    args = parser.parse_args()

    root = Path(args.directory).resolve()

    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    changed_files = []

    for py_file in root.rglob("*.py"):
        if fix_file(py_file, dry_run=args.dry_run, backup=not args.no_backup):
            changed_files.append(py_file)

    if args.dry_run:
        print("Archivos que se modificarían:")
    else:
        print("Archivos modificados:")

    for file in changed_files:
        print(file)

    print(f"\nTotal: {len(changed_files)} archivo(s)")


if __name__ == "__main__":
    main()
