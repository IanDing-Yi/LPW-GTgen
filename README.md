# LPW-GTgen

This repository generates pseudo-groundtruth image datasets from 3D Lincoln Pottery Works (LPW) vessel meshes and trains ceramic classification models on the resulting fragments. It supports two fragment-generation strategies: naive random-plane breaking and realistic fragments rendered from generated 3D meshes. The classifier can train on manually labeled fragments, pseudo-groundtruth fragments, or hybrid combinations of the two.

## Data availability and redistribution

This repository intentionally contains code, dependency information, and usage instructions only. It does not redistribute the 2D image data, 3D LPW vessel meshes, generated fragments, trained model weights, CSV manifests, or other derived LPW datasets because the repository authors do not control the data rights for those materials. Users must obtain any required source data from authorized data owners and confirm that their use complies with the applicable permissions and licenses.

## Workflow

1. Prepare clean, colorized vessel meshes.
2. Generate or repair 3D fragments with the scripts in the repository root.
3. Render fragment images and organize CSV manifests for the manual, naive, and realistic sets.
4. Train and evaluate the classifiers from `classifier/`.
5. Optionally create random evaluation splits or run Grad-CAM analysis.

The classifier expects CSV files without headers. Each row contains an image path and its class label. A typical ground-truth directory contains:

```text
groundtruth/
  manual_train.csv
  manual_validation.csv
  test.csv
  naive.csv
  realistic.csv
```

Image paths in the CSV files are resolved relative to the `base_path` passed to the classifier scripts.

## Requirements

The code is Python-based and uses PyTorch, pandas, NumPy, SciPy, scikit-image, scikit-learn, Pillow, tqdm, trimesh, pyrender, PyMeshLab, PyVista, PyWavefront, pycpd, Matplotlib, and the model dependencies used by `pretrain_model.py`. Install versions compatible with the CUDA/PyTorch installation on the host. Rendering also requires a working OpenGL or headless rendering environment.

Install the dependencies from [requirements.txt](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

For PyTorch, select a wheel compatible with the host's CPU/CUDA setup if the default pip resolution is not appropriate. Before running an experiment, verify that the selected Python environment can import the modules used by `classifier/train_test.py` and `rdm_3dbreak_trimesh.py`.

## Generating fragments

- `rdm_3dbreak_trimesh.py` — core mesh cleaning, breaking, alignment, and rendering functions.
- `fix_watertight.py` — mesh repair and watertightness helpers.
- `naive_gen_1000_per_class.py` — generates approximately 1,000 naive fragments per class and renders them.
- `naive_gen_1000_per_class_delay_render.py` — variant that renders existing naive fragments in a later pass.
- `realistic_gen_1000_per_class.py` — renders realistic fragments from generated mesh metadata.

The generation scripts require local paths to source meshes and output directories. Do not commit source LPW meshes, rendered images, generated fragments, manifests, trained weights, or logs unless you have explicit redistribution permission.

Generate naive fragments from a local text file containing one mesh path per line:

```bash
python naive_gen_1000_per_class.py \
  --mesh-list /path/to/local_meshes.txt \
  --fragment-output /path/to/generated_fragments \
  --render-output /path/to/rendered_images
```

Render existing naive fragments:

```bash
python naive_gen_1000_per_class_delay_render.py \
  --generated-mesh-root /path/to/generated_fragments \
  --original-mesh-root /path/to/source_meshes \
  --render-output /path/to/rendered_images \
  --log-dir /path/to/logs
```

Render realistic generated fragments from local mesh metadata:

```bash
python realistic_gen_1000_per_class.py \
  --generated-mesh-root /path/to/generated_mesh_metadata \
  --original-mesh-root /path/to/source_meshes \
  --render-output /path/to/rendered_images \
  --log-dir /path/to/logs
```

To repair an individual mesh:

```bash
python fix_watertight.py <input_file> <output_path>
```

## Classifier scripts

Run classifier scripts from the `classifier/` directory so their local imports resolve correctly.

- `train_test.py` — shared training, validation, testing, fine-tuning, random-crop, and ProtoNet routines.
- `pretrain_model.py` — loads the pretrained backbones used by the training routines.
- `gen_gt_runall.py` — trains on manual, naive, and realistic sets, then runs pseudo-groundtruth-to-manual and manual-to-pseudo-groundtruth hybrid fine-tuning.
- `gen_gt_runall_comb.py` — runs manual and combined (`comb.csv`) experiments and their hybrid variants.
- `gen_gt_runall_rand.py` — runs the manual, naive, and realistic experiments using a random evaluation split.
- `gen_gt_runall_comb_rand.py` — combined-set equivalent using a random evaluation split.
- `gen_gt_run_crop.py` — trains the manual-only model with random-crop augmentation.
- `gen_gt_protonet.py` — runs ProtoNet few-shot experiments using naive and realistic pseudo-groundtruth sets.
- `run_experiment.py` — argument-driven repeated experiments for the supported ConvNeXt backbones and balance settings.
- `hist_dataloader.py` — dataset and image transformation definitions used by the classifier.
- `prep_random_eval_set.py` — creates a random train/validation/test split from the manual CSVs.
- `prep_random_eval_set_balanced.py` — creates a class-balanced random split.
- `prep_random_eval_set_imbalanced.py` — creates an unbalanced random split while also handling pseudo-groundtruth CSVs.
- `gradcam_naive_realistic_analysis.py` — Grad-CAM analysis for naive and realistic models.
- `gradcam_comb_analysis.py` — Grad-CAM analysis for combined-set models.

The legacy training scripts use this positional form:

```bash
cd classifier
python gen_gt_runall.py <model> <base_path> <gt_path>
```

For example:

```bash
python gen_gt_runall.py convnext_base /path/to/local/image_root \
  groundtruth_swap_base_gt_rand_balance
```

The `base_path` is the root used to resolve image paths. The `gt_path` may be an absolute path or a path relative to the current working directory, depending on how the CSV manifests are organized.

## Repeated experiments

`run_experiment.py` uses datasets under `--dataset-root` or the `LPW_DATASET_ROOT` environment variable and writes each run under `experiment_outputs/`. Its default dataset naming convention is:

```text
{k}_groundtruth_swap_base_gt_rand_{balance}
```

Supported model names are `convnextv2_base` and `convnext_base`; supported balance values are `balance` and `imbalance`. The experiment names are `manual`, `naive`, `realistic`, `combined`, `naive_hybrid`, `realistic_hybrid`, `combined_hybrid`, `naive_reverse_hybrid`, `realistic_reverse_hybrid`, and `combined_reverse_hybrid`.

Example:

```bash
cd classifier
python run_experiment.py \
  --model convnext_base \
  --balance balance \
  --experiment naive_hybrid \
  --k 1 \
  --repeat 1 \
  --dataset-root /path/to/local/datasets
```

Each run expects `manual_train.csv`, `manual_validation.csv`, `test.csv`, `naive.csv`, and `realistic.csv` in the selected dataset directory. The combined CSV is created automatically when needed.

## Random evaluation splits

The basic splitter takes a source folder, output folder, per-class training size, and random seed:

```bash
cd classifier
python prep_random_eval_set.py \
  <base_path> <input_folder> <output_folder> <train_size> <random_seed>
```

For example:

```bash
python prep_random_eval_set.py "" groundtruth_swap groundtruth_random_eval 1000 7
```

The splitter keeps one validation image per class and places the remaining images in the test set. Existing output CSVs are moved into a numbered backup directory before new files are written.

## License

This project is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).