import os
import sys
from train_test import run_protonet

def run_all(model_name, gt_path, base_path):
    run_protonet(var_save_name=model_name + '_protonet_naive_outcomes', 
                 model_name=model_name, 
                 model_save_path=model_name + '_protonet_naive.pth',
                 train_csv=os.path.join(gt_path, 'naive.csv'),
                 anchor_csv=os.path.join(gt_path, 'manual_train.csv'),
                 valid_csv=os.path.join(gt_path, 'manual_validation.csv'),
                 test_csv=os.path.join(gt_path, 'test.csv'),
                 base_path=base_path,
                 run_count=1,
                 disp=True,
                 nb_cls=6, batch_size=10, lr=0.00001, patience=3, min_delta=0, max_episodes=1000)
                 
    run_protonet(var_save_name=model_name + '_protonet_realistic_outcomes', 
                 model_name=model_name, 
                 model_save_path=model_name + '_protonet_realistic.pth',
                 train_csv=os.path.join(gt_path, 'realistic.csv'),
                 anchor_csv=os.path.join(gt_path, 'manual_train.csv'),
                 valid_csv=os.path.join(gt_path, 'manual_validation.csv'),
                 test_csv=os.path.join(gt_path, 'test.csv'),
                 base_path=base_path,
                 run_count=1,
                 disp=True,
                 nb_cls=6, batch_size=10, lr=0.00001, patience=3, min_delta=0, max_episodes=1000)

def main():
    if len(sys.argv) != 4:
        print(sys.argv)
        print("Usage: python gen_gt_protonet.py <model> <base_path> <gt_path>")
        print("Example: python gen_gt_protonet.py protonet_convnext_base \"\" groundtruth")
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