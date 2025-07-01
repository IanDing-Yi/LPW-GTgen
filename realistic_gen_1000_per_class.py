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

log_path = 'G:/dhp_data/artifact_restore_identify/2d_render_realistic_gen_set/logs/'
gen_path = 'G:/dhp_data/artifact_restore_identify/2d_render_realistic_gen_set/'
base_path = 'G:/dhp_data/artifact_restore_identify/3d_model_gen_set/'
orig_path = 'G:/dhp_data/artifact_restore_identify/3d_model_clean_set/'
apendix = '128_metadata_mesh.txt'
# Find all txt files containing apendix in their name
matching_files = []
folder_name = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if apendix in file:
            matching_files.append(os.path.join(root, file))
            # Remove the need for folder name
            folder_name.append(file[:-len(apendix)-1])

executor = MultiThreadExecutor(wrap_render_existing_mesh)

print(f"Found {len(matching_files)} files:")
for idx, f in enumerate(matching_files):
    print(f"File {idx+1}: {f}")
    print(f"Folder name: {folder_name[idx]}")

    # Load the found file using pandas
    df = pd.read_csv(f, sep='\t', header=0)  # Adjust sep/header as needed
    # print(f"Loaded {len(df)} rows from {f}")
    
    orig_file = orig_path + folder_name[idx] + '.obj'
    meshes = []
    # Loop through rows
    for i, row in df.iterrows():
        meshes.append([i, row.values[4], orig_file, gen_path + folder_name[idx]])
        # print(meshes[-1])

    threads = 10
    num_fragments = len(meshes)
    print(f"Rendering {num_fragments} meshes with {threads} threads...")

    with ThreadPool(threads) as pool:
        for frag in tqdm(pool.imap(executor.opWrap, meshes), total=num_fragments):
            if isinstance(frag, str):
                with open(log_path + 'error_rendering.txt', 'a') as f:
                    f.write(frag + '\n')

