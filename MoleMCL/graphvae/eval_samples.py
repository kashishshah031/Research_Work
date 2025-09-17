import argparse, csv, os
from rdkit import Chem

def canon(smiles: str):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def read_smiles_csv(path: str, col: str):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        if col not in r.fieldnames:
            # fallback: first column
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
            return [row[0] for row in r if row]
        return [row[col] for row in r if row.get(col) is not None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="CSV with generated SMILES, col 'smiles'")
    ap.add_argument("--train_csv", required=True, help="Training dataset CSV with a 'smiles' column")
    ap.add_argument("--samples_col", default="smiles")
    ap.add_argument("--train_col", default="smiles")
    args = ap.parse_args()

    gen_raw = read_smiles_csv(args.samples, args.samples_col)
    train_raw = read_smiles_csv(args.train_csv, args.train_col)

    # Canonicalize
    gen_canon = [canon(s) for s in gen_raw]
    valid_canon = [s for s in gen_canon if s]
    train_canon_set = set(filter(None, (canon(s) for s in train_raw)))

    total = len(gen_raw)
    num_valid = len(valid_canon)
    validity = 100.0 * num_valid / max(1, total)

    unique_valid = len(set(valid_canon))
    uniqueness = 100.0 * unique_valid / max(1, num_valid)

    novel = [s for s in valid_canon if s not in train_canon_set]
    novelty = 100.0 * len(novel) / max(1, num_valid)

    print(f"Total: {total}")
    print(f"Valid: {num_valid} ({validity:.2f}%)")
    print(f"Uniqueness: {uniqueness:.2f}%")
    print(f"Novelty: {novelty:.2f}%")

if __name__ == "__main__":
    main()