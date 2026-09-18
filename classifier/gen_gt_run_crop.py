import os
import sys
from train_test import run, run_hybrid, run_random_crop

def run_all(model_name, gt_path, base_path):
    run_random_crop(model_name + '_manual_outcomes', model_name, model_name + '_manual.pth',
                    os.path.join(gt_path, 'manual_train.csv'),
                    os.path.join(gt_path, 'manual_validation.csv'),
                    os.path.join(gt_path, 'test.csv'),
                    base_path, 1, True,
                    nb_cls=6, batch_size=32, lr=0.0001, patience=5, min_delta=0, max_episodes=1000)

def main():
    if len(sys.argv) != 4:
        print("Usage: python gen_gt_runall.py <model> <base_path> <gt_path>")
        print("Example: python gen_gt_runall.py resnet50 \"\" groundtruth")
        sys.exit(1)
    
    model = sys.argv[1]
    base_path = sys.argv[2]
    gt_path = sys.argv[3]
    
    print(f"Running model: {model}")
    print(f"Base path: {base_path}")
    print(f"Ground truth path: {gt_path}")
    print("==============================================================")
    run_all(model, gt_path, base_path)
    print("==============================================================")

if __name__ == "__main__":
    main()