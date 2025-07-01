import os
import sys
import io

import numpy as np
import trimesh
import pymeshlab

from scipy.spatial import cKDTree

def print_mesh_info(ms):
    ms.set_selection_all()
    mesh = ms.current_mesh()
    selected_faces_count = mesh.selected_face_number()
    print("Number of faces of all:", selected_faces_count)
    ms.set_selection_none()

def convert_trimesh_to_meshlab(tm):
    """
    Convert a trimesh mesh to a pymeshlab MeshSet.
    This function takes a trimesh mesh and converts it to a pymeshlab MeshSet.
    """
    # Extract vertex and face data from the Trimesh object
    vertices = np.array(tm.vertices, dtype=np.float64)
    faces = np.array(tm.faces, dtype=np.int32)
    
    # Create a PyMeshLab Mesh object from the arrays
    ml_mesh = pymeshlab.Mesh(vertices, faces)

    # Create a MeshSet and add the mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(ml_mesh, "my_trimesh_converted")
    ms.set_current_mesh(0)

    return ms

def convert_meshlab_to_trimesh(ms):
    """
    Convert a pymeshlab MeshSet to a trimesh mesh.
    This function takes a pymeshlab MeshSet and converts it to a trimesh mesh.
    """
    current_ml_mesh = ms.current_mesh()

    # Extract vertex and face data from the PyMeshLab mesh
    vertices_back = current_ml_mesh.vertex_matrix().tolist()
    faces_back = current_ml_mesh.face_matrix().tolist()

    # Create a new Trimesh object
    tm = trimesh.Trimesh(vertices=vertices_back, faces=faces_back)

    return tm

def remove_small_disconnected_components(ms, verbose=False):
    """
    Remove small disconnected components from the mesh.
    This function will iteratively remove small disconnected components
    until no more small components are found.
    """
    ms.compute_selection_by_small_disconnected_components_per_face()

    selected_faces_count = ms.current_mesh().selected_face_number()
    if verbose:
        print("Number of faces found in small disconnected components:", selected_faces_count)

    while selected_faces_count > 0:

        ms.meshing_remove_selected_faces()

        ms.set_selection_none()
        ms.compute_selection_by_small_disconnected_components_per_face()

        selected_faces_count = ms.current_mesh().selected_face_number()
        if verbose:
            print("Number of faces found in small disconnected components:", selected_faces_count)

def _repair_filters():
    """
    Return a list of mesh repair filters.
    This function returns a list of filter names that can be used for mesh repair operations.
    """
    return [
        'meshing_merge_close_vertices',
        'meshing_remove_connected_component_by_diameter',
        'meshing_remove_duplicate_faces',
        'meshing_remove_duplicate_vertices',
        'meshing_remove_unreferenced_vertices',
        'meshing_repair_non_manifold_edges',
    ]

def repair_mesh(ms, verbose=False):
    """
    Perform mesh repair operations.
    This function will apply various mesh repair operations to the current mesh.
    """
    vertices_before = ms.current_mesh().vertex_number()
    for filter_name in _repair_filters():
        if verbose:
            print(f"Applying filter: {filter_name}")
        ms.apply_filter(filter_name)
    vertices_after = ms.current_mesh().vertex_number()
    merged_vertices = vertices_before - vertices_after
    if verbose:
        print(f"Vertices before repair: {vertices_before}, after repair: {vertices_after}")
        print(f"Number of vertices changed during repair: {merged_vertices}")
    while merged_vertices > 0:
        vertices_before = ms.current_mesh().vertex_number()
        for filter_name in _repair_filters():
            if verbose:
                print(f"Applying filter: {filter_name}")
            ms.apply_filter(filter_name)
        vertices_after = ms.current_mesh().vertex_number()
        merged_vertices = vertices_before - vertices_after
        if verbose:
            print(f"Vertices before repair: {vertices_before}, after repair: {vertices_after}")
            print(f"Number of vertices changed during repair: {merged_vertices}")

def clean_trimesh(mesh):
    # # Split and keep largest component
    # components = mesh.split(only_watertight=False)
    # mesh = max(components, key=lambda m: m.faces.shape[0])

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.process(validate=True)
    return mesh

def clean_mesh(ms, need_convert=False, verbose=False):
    """
    Clean the mesh by removing small disconnected components and applying repair filters.
    This function will first remove small disconnected components and then apply a series of repair filters.
    """
    if need_convert:
        if verbose:
            print("Converting trimesh to pymeshlab MeshSet.")
        orig_vertices = np.array(ms.vertices)
        orig_colors = np.array(ms.visual.vertex_colors)
        ms = convert_trimesh_to_meshlab(ms)
        

    remove_small_disconnected_components(ms, verbose)
    repair_mesh(ms, verbose)
    if verbose:
        print("Mesh cleaned.")
        print_mesh_info(ms)

    if need_convert:
        if verbose:
            print("Converting pymeshlab MeshSet back to trimesh.")
        ms = convert_meshlab_to_trimesh(ms)
        tree = cKDTree(orig_vertices)
        dists, idxs = tree.query(ms.vertices)
        ms.visual.vertex_colors = orig_colors[idxs]

    return ms

def run(input_file, output_path, verbose=False):
    """
    Main function to run the mesh processing.
    This function initializes a MeshSet, loads a mesh, and applies various processing steps.
    """
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(input_file)

    if verbose:
        print(f"Loaded mesh from {input_file}")
        print_mesh_info(ms)

    remove_small_disconnected_components(ms, verbose)

    if verbose:
        print("Removed small disconnected components.")
        print_mesh_info(ms)

    repair_mesh(ms, verbose)

    if verbose:
        print("Applied mesh repair filters.")
        print_mesh_info(ms)

    # Close holes in the mesh and check if any faces were merged
    faces_before = ms.current_mesh().face_number()
    ms.meshing_close_holes()
    faces_after = ms.current_mesh().face_number()
    merged_faces = abs(faces_before - faces_after)
    if verbose:
        print(f"Vertices before repair: {faces_before}, after repair: {faces_after}")
        print(f"Number of vertices changed during repair: {merged_faces}")
    while merged_faces > 0:
        repair_mesh(ms, verbose)
        faces_before = ms.current_mesh().face_number()
        ms.meshing_close_holes()
        faces_after = ms.current_mesh().face_number()
        merged_faces = abs(faces_before - faces_after)
        if verbose:
            print(f"Vertices before repair: {faces_before}, after repair: {faces_after}")
            print(f"Number of vertices changed during repair: {merged_faces}")

    if verbose:
        print("Closed holes in the mesh.")
        print_mesh_info(ms)

    ms.generate_surface_reconstruction_screened_poisson()

    if verbose:
        print("Generated surface reconstruction using screened Poisson.")
        print_mesh_info(ms)

    # Save the processed mesh
    output_path = output_path
    ms.save_current_mesh(output_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(f"Argument: {arg}")
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_path = sys.argv[2]
        run(input_file, output_path, verbose=False)
    elif len(sys.argv) == 4:
        if sys.argv[1] == '--verbose':
            verbose = True
            print("Running in verbose mode.")
        else:
            verbose = False
        input_file = sys.argv[2]
        output_path = sys.argv[3]
        run(input_file, output_path, verbose=verbose)
    elif len(sys.argv) == 2:
        if sys.argv[1] == '--test':
            verbose = True
            input_file = 'G:/dhp_data/artifact_restore_identify/3d_model_test/11100-12-1_BeanPot_OBJ_Decimated.obj'
            output_path = 'G:/dhp_data/artifact_restore_identify/3d_model_test/11100-12-1_BeanPot_OBJ_Decimated_fix_watertight.obj'
            run(input_file, output_path, verbose=True)
    else:
        print("No arguments provided.")
        print("Usage: python script.py <input_file> <output_path>")
        print("For more verbosity: python script.py --verbose <input_file> <output_path>")
