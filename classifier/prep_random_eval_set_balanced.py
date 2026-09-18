import os
import sys

import pandas as pd


SPLIT_FILES = ("manual_train.csv", "manual_validation.csv", "test.csv")


def prep_random_eval_set_balanced(
    base_path, gt_input_folder, gt_output_folder, train_size, random_seed=7
):
    input_folder = os.path.join(base_path, gt_input_folder)
    output_folder = os.path.join(base_path, gt_output_folder)
    combined_df = pd.concat(
        [pd.read_csv(os.path.join(input_folder, file_name), header=None) for file_name in SPLIT_FILES],
        ignore_index=True,
    )
    label_column = combined_df.columns[-1]
    class_counts = combined_df[label_column].value_counts()
    required_per_class = train_size + 1
    insufficient_classes = class_counts[class_counts < required_per_class]
    if not insufficient_classes.empty:
        raise ValueError(
            f"Each class needs at least {required_per_class} images; insufficient classes: "
            f"{insufficient_classes.to_dict()}"
        )

    shuffled_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    class_order = shuffled_df.groupby(shuffled_df.iloc[:, -1]).cumcount()
    train_df = shuffled_df.loc[class_order < train_size]
    validation_df = shuffled_df.loc[class_order == train_size]
    test_df = shuffled_df.loc[class_order > train_size]

    os.makedirs(output_folder, exist_ok=True)
    train_df.to_csv(os.path.join(output_folder, "manual_train.csv"), index=False, header=False)
    validation_df.to_csv(os.path.join(output_folder, "manual_validation.csv"), index=False, header=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False, header=False)

    print(
        f"Created balanced random eval set in {output_folder}: "
        f"{train_size} train and 1 validation image per class."
    )
    for class_label in sorted(class_counts.index):
        print(
            f"Class {class_label}: "
            f"train={len(train_df[train_df[label_column] == class_label])}, "
            f"validation={len(validation_df[validation_df[label_column] == class_label])}, "
            f"test={len(test_df[test_df[label_column] == class_label])}"
        )


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python prep_random_eval_set_balanced.py "
            "<base_path> <gt_input_folder> <gt_output_folder> "
            "<train_size_per_class> <random_seed>"
        )
        sys.exit(1)

    prep_random_eval_set_balanced(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]),
    )


if __name__ == "__main__":
    main()