"""Ponto de entrada: reproduz todos os experimentos e gráficos do trabalho.

Uso:

    python run_all.py                 # baixa a coleção (se preciso) e roda tudo
    python run_all.py --no-download   # usa os arquivos já presentes em data/cran
    python run_all.py --skip-figures  # só os resultados numéricos

Saídas:
    data/cran/   -- coleção Cranfield (documentos, consultas, qrels)
    results/     -- CSVs e JSONs com todos os resultados, por consulta e agregados
    figures/     -- figuras do relatório em PNG
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ir_tp1.dataset import load_cranfield  # noqa: E402
from ir_tp1.experiments import run_all_experiments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="não baixar a coleção; usar os arquivos já em data/cran",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="não gerar as figuras"
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    args = parser.parse_args()

    started = time.time()

    print("Carregando a coleção Cranfield...")
    collection = load_cranfield(args.data_dir, download=not args.no_download)
    print(
        f"  {len(collection.documents)} documentos, "
        f"{len(collection.queries)} consultas, "
        f"{sum(len(v) for v in collection.qrels.values())} julgamentos"
    )

    results = run_all_experiments(collection, Path(args.data_dir), Path(args.results_dir))

    if not args.skip_figures:
        print("Gerando figuras...")
        from ir_tp1.plots import generate_all

        for path in generate_all(results, Path(args.figures_dir)):
            print(f"      {path.name}")

    aggregates = results["metadata"]["aggregates"]
    print("\n=== Resultado agregado (225 consultas) ===")
    header = f"{'modelo':<6}" + "".join(f"{m:>10}" for m in aggregates["BM25"])
    print(header)
    for model, values in aggregates.items():
        print(f"{model:<6}" + "".join(f"{v:>10.4f}" for v in values.values()))

    print(f"\nConcluído em {time.time() - started:.1f}s.")
    print(f"Resultados em {args.results_dir}/ e figuras em {args.figures_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
