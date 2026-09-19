import argparse
import os

from tqdm import tqdm
import rdm_3dbreak_trimesh as break3d
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd

# thread pool executor
class MultiThreadExecutor():
    def __init__(self, op):
        self.op = op
    
    def opWrap(self, params):
        # Initialization
        droppingInds = self.op(params)
        return droppingInds

    def calculateParallel(self, paraList, threads=2):
        pool = ThreadPool(threads)
        droppingInds = pool.map(self.opWrap, paraList)
        pool.close()
        pool.join()
        return droppingInds

def wrap_render_existing_mesh(params):
    obj_file, original_mesh, render_savepath = params
    img = break3d.render_existing_mesh_naive(obj_file, original_mesh, 
                                             save_render = True, render_savepath=render_savepath, 
                                             verbose=False, visualize=False)
    # Retry 10 times if rendering fails
    retry_count = 0
    max_retries = 30
    while img is None and retry_count < max_retries:
        retry_count += 1
        print(f"Retry {retry_count}/{max_retries} for {obj_file}...")
        img = break3d.render_existing_mesh_naive(obj_file, original_mesh, 
                                                save_render = True, render_savepath=render_savepath, 
                                                verbose=False, visualize=False)
        
    if img is None:
        return f"{obj_file}"
    return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-mesh-root", required=True, help="Root containing generated naive mesh fragments and all.txt.")
    parser.add_argument("--original-mesh-root", required=True, help="Root containing original meshes named by fragment folder.")
    parser.add_argument("--render-output", required=True, help="Directory for rendered fragment images.")
    parser.add_argument("--log-dir", required=True, help="Directory for rendering error logs.")
    parser.add_argument("--threads", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.render_output, exist_ok=True)

    with open(os.path.join(args.generated_mesh_root, 'all.txt'), 'r') as f:
        all_files = f.read().splitlines()

    file_groups = {}
    for file in all_files:
        last_folder = os.path.basename(os.path.normpath(file[:file.rfind('/')]))
        if last_folder not in file_groups:
            file_groups[last_folder] = []
        file_groups[last_folder].append([
            file,
            os.path.join(args.original_mesh_root, last_folder + '.obj'),
            os.path.join(args.render_output, last_folder),
        ])

    executor = MultiThreadExecutor(wrap_render_existing_mesh)

    for folder_name in file_groups:
        print(f"Processing folder: {folder_name}")
        meshes = file_groups[folder_name]
        num_fragments = len(meshes)
        print(f"Rendering {num_fragments} meshes in folder '{folder_name}' with {args.threads} threads...")

        with ThreadPool(args.threads) as pool:
            for frag in tqdm(pool.imap(executor.opWrap, meshes), total=num_fragments):
                if isinstance(frag, str):
                    with open(os.path.join(args.log_dir, 'error_rendering.txt'), 'a') as f:
                        f.write(frag + '\n')


if __name__ == "__main__":
    main()
