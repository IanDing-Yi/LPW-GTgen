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

log_path = 'G:/dhp_data/artifact_restore_identify/2d_render_naive_gen_delay_render_set/logs/'
gen_path = 'G:/dhp_data/artifact_restore_identify/2d_render_naive_gen_delay_render_set/'
base_path = 'G:/dhp_data/artifact_restore_identify/3d_model_naive_gen_set/'
orig_path = 'G:/dhp_data/artifact_restore_identify/3d_model_clean_set/'

with open(os.path.join(base_path, 'all.txt'), 'r') as f:
    all_files = f.read().splitlines()

file_groups = {}
for file in all_files:
    last_folder = os.path.basename(os.path.normpath(file[:file.rfind('/')]))
    # print(f"Processing file: {file}, last folder: {last_folder}")
    # break
    if last_folder not in file_groups:
        file_groups[last_folder] = []
    file_groups[last_folder].append([file, orig_path + last_folder + '.obj', gen_path + last_folder])

executor = MultiThreadExecutor(wrap_render_existing_mesh)

for folder_name in file_groups:
    print(f"Processing folder: {folder_name}")
    meshes = file_groups[folder_name]
    
    threads = 10
    num_fragments = len(meshes)
    print(f"Rendering {num_fragments} meshes in folder '{folder_name}' with {threads} threads...")

    with ThreadPool(threads) as pool:
        for frag in tqdm(pool.imap(executor.opWrap, meshes), total=num_fragments):
            if isinstance(frag, str):
                with open(log_path + 'error_rendering.txt', 'a') as f:
                    f.write(frag + '\n')
