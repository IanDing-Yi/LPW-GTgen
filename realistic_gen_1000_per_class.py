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
    obj_file, percentage, original_mesh, render_savepath = params
    img = break3d.render_existing_mesh(obj_file, percentage, original_mesh, 
                                       save_render = True, render_savepath=render_savepath, 
                                       verbose=False, visualize=False)
    # Retry 10 times if rendering fails
    retry_count = 0
    max_retries = 30
    while img is None and retry_count < max_retries:
        retry_count += 1
        print(f"Retry {retry_count}/{max_retries} for {obj_file}...")
        img = break3d.render_existing_mesh(obj_file, percentage, original_mesh, 
                                           save_render = True, render_savepath=render_savepath, 
                                           verbose=False, visualize=False)
        
    if img is None:
        return f"{obj_file},{percentage}"
    return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-mesh-root", required=True, help="Root containing generated mesh metadata files.")
    parser.add_argument("--original-mesh-root", required=True, help="Root containing original meshes named by metadata folder.")
    parser.add_argument("--render-output", required=True, help="Directory for rendered fragment images.")
    parser.add_argument("--log-dir", required=True, help="Directory for rendering error logs.")
    parser.add_argument("--metadata-suffix", default="128_metadata_mesh.txt")
    parser.add_argument("--threads", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.render_output, exist_ok=True)
    matching_files = []
    folder_name = []
    for root, dirs, files in os.walk(args.generated_mesh_root):
        for file in files:
            if args.metadata_suffix in file:
                matching_files.append(os.path.join(root, file))
                folder_name.append(file[:-len(args.metadata_suffix)-1])

    executor = MultiThreadExecutor(wrap_render_existing_mesh)

    print(f"Found {len(matching_files)} files:")
    for idx, f in enumerate(matching_files):
        print(f"File {idx+1}: {f}")
        print(f"Folder name: {folder_name[idx]}")

        df = pd.read_csv(f, sep='\t', header=0)
        orig_file = os.path.join(args.original_mesh_root, folder_name[idx] + '.obj')
        meshes = []
        for i, row in df.iterrows():
            meshes.append([i, row.values[4], orig_file, os.path.join(args.render_output, folder_name[idx])])

        num_fragments = len(meshes)
        print(f"Rendering {num_fragments} meshes with {args.threads} threads...")

        with ThreadPool(args.threads) as pool:
            for frag in tqdm(pool.imap(executor.opWrap, meshes), total=num_fragments):
                if isinstance(frag, str):
                    with open(os.path.join(args.log_dir, 'error_rendering.txt'), 'a') as f:
                        f.write(frag + '\n')


if __name__ == "__main__":
    main()

