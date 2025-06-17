import argparse
import yaml
from pathlib import Path
from sklearn.model_selection import KFold

# by : python make_fold.py --data_yaml data/kaist-rgbt.yaml --folds 5 

def make_fold_yaml(data_yaml: str, folds: int, seed: int = 0) -> str:
    """
    Create train/val split files and a fold-specific YAML from a base dataset YAML.

    Returns the path to the new fold YAML.
    """
    # 1) Load base yaml
    base = yaml.safe_load(open(data_yaml, 'r', encoding='utf-8'))
    path = Path(base['path'])

    # 2) Read full train list
    train_list = path / base['train'][0]
    lines = train_list.read_text().splitlines()

    # 3) KFold split (take only first fold)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train_idx, val_idx = next(kf.split(lines))

    # 4) Write new txt files
    t_txt = path / f"{train_list.stem}_fold{folds}.txt"
    v_txt = path / f"{Path(base['val'][0]).stem}_fold{folds}.txt"
    t_txt.write_text("\n".join([lines[i] for i in train_idx]) + "\n")
    v_txt.write_text("\n".join([lines[i] for i in val_idx]) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fold-specific YOLOv5 data YAML')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to base dataset YAML')
    parser.add_argument('--folds', type=int, default=2, help='Number of folds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    make_fold_yaml(args.data_yaml, args.folds, args.seed)
