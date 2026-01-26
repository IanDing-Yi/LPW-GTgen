import os
import sys
from train_test import run, run_hybrid
# [1.46993504, 1.83937636, 1.63301425, 1.10534349, 1., 1.]
def run_all(model_name, gt_path, base_path):
    run(gt_path + '_' + model_name + '_manual_outcomes',
        model_name,
        gt_path + '_' + model_name + '_manual.pth',
        os.path.join(gt_path, 'manual_train.csv'),
        os.path.join(gt_path, 'manual_validation.csv'),
        os.path.join(gt_path, 'test.csv'),
        base_path, 1, True,
        nb_cls=5, batch_size=32, lr=0.0001, patience=5, min_delta=0, max_episodes=1000,
        cls_weights=[1., 1., 1., 1., 1.])
    
    run(gt_path + '_' + model_name + '_comb_outcomes',
        model_name, 
        gt_path + '_' + model_name + '_comb.pth',
        os.path.join(gt_path, 'comb.csv'),
        os.path.join(gt_path, 'manual_validation.csv'),
        os.path.join(gt_path, 'test.csv'),
        base_path, 1, True,
        nb_cls=5, batch_size=32, lr=0.0001, patience=5, min_delta=0, max_episodes=1000,
        cls_weights=[1., 1., 1., 1., 1.])
    
    run_hybrid(gt_path + '_' + model_name + '_comb_hybrid_outcomes',
               model_name,
               gt_path + '_' + model_name + '_manual.pth',
               gt_path + '_' + model_name + '_naive_hybrid.pth',
               os.path.join(gt_path, 'comb.csv'),
               os.path.join(gt_path, 'manual_validation.csv'),
               os.path.join(gt_path, 'test.csv'),
               base_path, 1, True,
               nb_cls=5, batch_size=32, lr=0.0001, patience=5, min_delta=0, max_episodes=1000,
               cls_weights=[1., 1., 1., 1., 1.])
    
    run_hybrid(gt_path + '_' + model_name + '_comb_hybrid_reverse_outcomes',
               model_name,
               gt_path + '_' + model_name + '_comb.pth',
               gt_path + '_' + model_name + '_comb_hybrid_reverse.pth',
               os.path.join(gt_path, 'manual_train.csv'),
               os.path.join(gt_path, 'manual_validation.csv'),
               os.path.join(gt_path, 'test.csv'),
               base_path, 1, True,
               nb_cls=5, batch_size=32, lr=0.0001, patience=5, min_delta=0, max_episodes=1000,
               cls_weights=[1., 1., 1., 1., 1.])
               
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