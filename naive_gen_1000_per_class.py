import argparse

import rdm_3dbreak_trimesh as break3d

verbose = False
visualize = False
save_frag = True
save_render = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-list", required=True, help="Text file containing one input mesh path per line.")
    parser.add_argument("--fragment-output", required=True, help="Directory for generated 3D fragments.")
    parser.add_argument("--render-output", required=True, help="Directory for rendered 2D fragment images.")
    parser.add_argument("--fragments-per-mesh", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=14)
    return parser.parse_args()


def load_mesh_list(mesh_list):
    with open(mesh_list, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    args = parse_args()
    obj_files = load_mesh_list(args.mesh_list)

    for f in obj_files:
        break3d.run_parallel(
            f,
            args.fragments_per_mesh,
            args.workers,
            verbose,
            visualize,
            save_frag,
            args.fragment_output,
            save_render,
            args.render_output,
        )