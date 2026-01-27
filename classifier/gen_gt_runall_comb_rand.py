import os
import sys
import shutil
from train_test import run, run_hybrid
from prep_random_eval_set import prep_random_eval_set

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
    if len(sys.argv) != 7:
        print("Usage: python gen_gt_runall.py <model> <base_path> <gt_path> <train_size> <rand_gt_path> <nb_repeats>")
        print("Example: python gen_gt_runall.py resnet50 \"\" groundtruth")
        sys.exit(1)
    
    model = sys.argv[1]
    base_path = sys.argv[2]
    gt_path = sys.argv[3]
    train_size = sys.argv[4]
    rand_gt_path = f"{train_size}_{gt_path}_{model}_{sys.argv[5]}"
    nb_repeats = int(sys.argv[6])

    for repeat_idx in range(nb_repeats):
        print(f"=================== Repeat {repeat_idx+1} / {nb_repeats} ===================")
        # prepare random eval set
        prep_random_eval_set(base_path, gt_path, rand_gt_path, int(train_size), random_seed=repeat_idx+1)

        
        print(f"Running model: {model}")
        print(f"Base path: {base_path}")
        print(f"Ground truth path: {rand_gt_path}")
        print("==============================================================")
        run_all(model, rand_gt_path, base_path)
        print("==============================================================")

        # backup .pkl files
        # 'result_data_' + rand_gt_path + '_' + model_name + <suffix> + '_0' + '.pkl'
            
        suffixes = ['_manual_outcomes', '_comb_outcomes', '_comb_hybrid_outcomes', '_comb_hybrid_reverse_outcomes']
        for suffix in suffixes:
            src_pkl = f"result_data_{rand_gt_path}_{model}{suffix}_0.pkl"
            # count existing backup files and create a new backup name
            backup_folder = os.path.join(base_path, f"result_data_{rand_gt_path}_{model}{suffix}")
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)
            # count existing files in backup folder
            existing_files = [f for f in os.listdir(backup_folder) if f.startswith(f"result_data_{rand_gt_path}_{model}{suffix}_0_")]
            next_index = len(existing_files) + 1
            dst_pkl = os.path.join(backup_folder, f"result_data_{rand_gt_path}_{model}{suffix}_0_{next_index}.pkl")
            if os.path.exists(os.path.join(base_path, src_pkl)):
                shutil.move(src_pkl, dst_pkl)
                print(f"Backed up outcome file {src_pkl} to {dst_pkl}_{next_index}.pkl")

if __name__ == "__main__":
    main()