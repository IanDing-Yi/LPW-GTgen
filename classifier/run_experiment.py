import argparse
import os

import pandas as pd

from train_test import run, run_hybrid


DEFAULT_DATASET_ROOT = os.environ.get("LPW_DATASET_ROOT")
DATASET_PREFIX = "{k}_groundtruth_swap_base_gt_rand_{balance}"
EXPERIMENTS = {
    "manual": "manual",
    "naive": "naive",
    "realistic": "realistic",
    "combined": "combined",
    "naive_hybrid": "naive_hybrid",
    "realistic_hybrid": "realistic_hybrid",
    "combined_hybrid": "combined_hybrid",
    "naive_reverse_hybrid": "naive_reverse_hybrid",
    "realistic_reverse_hybrid": "realistic_reverse_hybrid",
    "combined_reverse_hybrid": "combined_reverse_hybrid",
}


def make_combined_csv(dataset_dir):
    combined_path = os.path.join(dataset_dir, "combined.csv")
    if not os.path.exists(combined_path):
        naive = pd.read_csv(os.path.join(dataset_dir, "naive.csv"), header=None)
        realistic = pd.read_csv(os.path.join(dataset_dir, "realistic.csv"), header=None)
        temporary_path = f"{combined_path}.{os.getpid()}.tmp"
        pd.concat([naive, realistic], ignore_index=True).to_csv(
            temporary_path, index=False, header=False
        )
        os.replace(temporary_path, combined_path)
    return combined_path


def train_once(experiment, model, dataset_dir, image_root, output_prefix, repeat):
    validation_csv = os.path.join(dataset_dir, "manual_validation.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    manual_csv = os.path.join(dataset_dir, "manual_train.csv")
    pseudo_csv = {
        "naive": os.path.join(dataset_dir, "naive.csv"),
        "realistic": os.path.join(dataset_dir, "realistic.csv"),
        "combined": make_combined_csv(dataset_dir),
    }
    common = dict(
        base_path=image_root,
        run_count=1,
        disp=True,
        nb_cls=5,
        batch_size=32,
        lr=0.0001,
        patience=5,
        min_delta=0,
        max_episodes=1000,
        cls_weights=[1., 1., 1., 1., 1.],
    )

    def train(train_csv, model_suffix):
        model_path = f"{output_prefix}_repeat{repeat}_{model_suffix}.pth"
        result_name = f"{output_prefix}_{model_suffix}_repeat{repeat}"
        run(
            result_name,
            model,
            model_path,
            train_csv,
            validation_csv,
            test_csv,
            **common,
        )
        return model_path

    def fine_tune(pretrained_path, train_csv, model_suffix):
        model_path = f"{output_prefix}_repeat{repeat}_{model_suffix}.pth"
        result_name = f"{output_prefix}_{model_suffix}_repeat{repeat}"
        run_hybrid(
            result_name,
            model,
            pretrained_path,
            model_path,
            train_csv,
            validation_csv,
            test_csv,
            **common,
        )

    if experiment == "manual":
        train(manual_csv, "manual")
        return

    if experiment in ("naive", "realistic", "combined"):
        train(pseudo_csv[experiment], experiment)
        return

    if experiment.endswith("_hybrid") and "reverse" not in experiment:
        pseudo_name = experiment.removesuffix("_hybrid")
        pretrained = train(pseudo_csv[pseudo_name], pseudo_name)
        fine_tune(pretrained, manual_csv, experiment)
        return

    pseudo_name = experiment.removesuffix("_reverse_hybrid")
    pretrained = train(manual_csv, "manual")
    fine_tune(pretrained, pseudo_csv[pseudo_name], experiment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convnextv2_base", "convnext_base"), required=True)
    parser.add_argument("--balance", choices=("balance", "imbalance"), required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--k", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 21), required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, required=DEFAULT_DATASET_ROOT is None)
    args = parser.parse_args()

    dataset_name = DATASET_PREFIX.format(k=args.k, balance=args.balance)
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(dataset_dir)
    output_prefix = f"{dataset_name}_{args.model}_{args.experiment}"
    output_dir = os.path.join(
        dataset_root,
        "experiment_outputs",
        dataset_name,
        args.model,
        args.experiment,
        f"repeat_{args.repeat}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Running {args.experiment}: model={args.model}, "
        f"dataset={dataset_name}, repeat={args.repeat}/20, output={output_dir}"
    )
    previous_directory = os.getcwd()
    os.chdir(output_dir)
    try:
        train_once(
            args.experiment,
            args.model,
            dataset_dir,
            dataset_root,
            output_prefix,
            args.repeat,
        )
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()