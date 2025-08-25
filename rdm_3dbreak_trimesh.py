import os
import sys
import time
import pymeshlab
import trimesh
import numpy as np
import pyrender
import pywavefront
# import open3d as o3d
# from vedo import Mesh, show, Light
import pyvista as pv
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation as R
from pycpd import RigidRegistration

from fix_watertight import clean_trimesh, clean_mesh

from multiprocessing.dummy import Pool as ThreadPool

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

def random_plane(mesh, debug=False, seed=None):
    if debug:
        if seed is None:
            raise ValueError("Seed must be provided for debug mode")
        np.random.seed(seed)
    # Get mesh bounds
    bounds = mesh.bounds
    center = mesh.centroid
    # Random normal
    normal = np.random.randn(3)
    normal /= np.linalg.norm(normal)
    # Random point near the center
    point = center + (np.random.rand(3) - 0.5) * (bounds[1] - bounds[0]) * 0.2
    return point, normal

def break_mesh(mesh, verbose=False, debug=False):
    try:
        part = mesh.copy()
        rand_int = np.random.randint(3)
        for _ in range(rand_int+1):
            if debug:
                point, normal = random_plane(part, debug=True, seed=124)
            else:
                point, normal = random_plane(part)
            section = part.section(plane_origin=point, plane_normal=normal)
            # If section exists, create a mesh on the intersection plane
            if section is not None and len(section.entities) > 0:
                # Get the 3D points of the intersection curve
                curves = section.discrete
                section_points = max(curves, key=lambda x: len(x))
                if len(section_points) > 2:
                    # Project points onto the plane (2D)
                    # Find two orthogonal vectors on the plane
                    v1 = np.cross(normal, [1, 0, 0])
                    if np.linalg.norm(v1) < 1e-6:
                        v1 = np.cross(normal, [0, 1, 0])
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = np.cross(normal, v1)
                    v2 = v2 / np.linalg.norm(v2)
                    # Build 2D coordinates in the plane
                    local_coords = np.dot(section_points - point, np.vstack([v1, v2]).T)
                    # Triangulate the 2D polygon
                    
                    tri = Delaunay(local_coords)
                    # Create mesh in 3D
                    plane_mesh = trimesh.Trimesh(vertices=section_points, faces=tri.simplices, process=False)

            # Split mesh by plane
            # print('before', part.visual.vertex_colors)
            _part = part.slice_plane(plane_origin=point, plane_normal=normal)
            orig_colors = part.visual.vertex_colors
            new_colors = np.zeros((len(_part.vertices), orig_colors.shape[1]), dtype=orig_colors.dtype)
            # For each new vertex, find the closest original vertex (simple nearest neighbor)
            kdtree = cKDTree(part.vertices)
            dists, indices = kdtree.query(_part.vertices)
            new_colors = orig_colors[indices]
            _part.visual.vertex_colors = new_colors # assign colors
            part = _part # update part
            # print('after', part.visual.vertex_colors)
            part = trimesh.util.concatenate([part, plane_mesh])

            if verbose:
                print(f"Cut fragment with plane at point {point} and normal {normal}")
                print(part)
        if verbose:
            print(f"Fragment created with {len(part.vertices)} vertices and {len(part.faces)} faces.")
        if part:
            # rot_matrix = rot_to_plane(normal)
            # Move the center of the fragment to (0, 0, 0)
            translation = -part.centroid
            part.apply_translation(translation)
            # Apply rotation to align with the camera view
            # plane_mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))

            return part

    except Exception as e:
        print(e)
        return None
    
    return None

def trans_mesh_colors(orimesh_path, newmesh_path):
    # Load original mesh
    orig_mesh = trimesh.load(orimesh_path, force='mesh')
    orig_vertices = np.array(orig_mesh.vertices)
    orig_colors = orig_mesh.visual.to_color().vertex_colors
    
    # Load new mesh
    new_mesh = trimesh.load(newmesh_path, force='mesh')
    
    # Create a KDTree for fast nearest neighbor search
    tree = cKDTree(orig_vertices)
    
    # Find nearest neighbors in the original mesh for each vertex in the new mesh
    dists, idxs = tree.query(new_mesh.vertices)
    
    # Assign colors from the original mesh to the new mesh
    new_mesh.visual.vertex_colors = orig_colors[idxs]
    
    return new_mesh

def break_mesh_delay_concat(mesh, verbose=False, debug=False):
    try:
        part = mesh.copy()
        rand_int = np.random.randint(3)
        plane_mesh_fills = []
        for _ in range(rand_int+1):
            if debug:
                point, normal = random_plane(part, debug=True, seed=124)
            else:
                point, normal = random_plane(part)
            section = part.section(plane_origin=point, plane_normal=normal)
            # If section exists, create a mesh on the intersection plane
            if section is not None and len(section.entities) > 0:
                # Get the 3D points of the intersection curve
                curves = section.discrete
                section_points = max(curves, key=lambda x: len(x))
                if len(section_points) > 2:
                    # Project points onto the plane (2D)
                    # Find two orthogonal vectors on the plane
                    v1 = np.cross(normal, [1, 0, 0])
                    if np.linalg.norm(v1) < 1e-6:
                        v1 = np.cross(normal, [0, 1, 0])
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = np.cross(normal, v1)
                    v2 = v2 / np.linalg.norm(v2)
                    # Build 2D coordinates in the plane
                    local_coords = np.dot(section_points - point, np.vstack([v1, v2]).T)
                    # Triangulate the 2D polygon
                    
                    tri = Delaunay(local_coords)
                    # Create mesh in 3D
                    plane_mesh = trimesh.Trimesh(vertices=section_points, faces=tri.simplices, process=False)

            # Split mesh by plane
            part = part.slice_plane(plane_origin=point, plane_normal=normal)

            for _id in range(len(plane_mesh_fills)):
                plane_mesh_fills[_id] = plane_mesh_fills[_id].slice_plane(plane_origin=point, plane_normal=normal)
            
            plane_mesh_fills.append(plane_mesh)

            if verbose:
                print(f"Cut fragment with plane at point {point} and normal {normal}")
                print(part)
        if verbose:
            print(f"Fragment created with {len(part.vertices)} vertices and {len(part.faces)} faces.")
        if part:
            # rot_matrix = rot_to_plane(normal)
            # Move the center of the fragment to (0, 0, 0)
            translation = -part.centroid
            part.apply_translation(translation)
            # Apply rotation to align with the camera view
            # plane_mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))

            for _id in range(len(plane_mesh_fills)):
                plane_mesh_fills[_id].apply_translation(translation)

            return part, plane_mesh_fills

    except Exception as e:
        print(e)

    return None, None

def old_break_mesh(mesh, num_cuts=3, verbose=False, debug=False):
    fragments = []
    for it in range(num_cuts):
        try:
            part = mesh.copy()
            rand_int = np.random.randint(3)
            for _ in range(rand_int+1):
                if debug:
                    point, normal = random_plane(part, debug=True, seed=124+it)
                else:
                    point, normal = random_plane(part)
                section = part.section(plane_origin=point, plane_normal=normal)
                # If section exists, create a mesh on the intersection plane
                if section is not None and len(section.entities) > 0:
                    # Get the 3D points of the intersection curve
                    curves = section.discrete
                    section_points = max(curves, key=lambda x: len(x))
                    if len(section_points) > 2:
                        # Project points onto the plane (2D)
                        # Find two orthogonal vectors on the plane
                        v1 = np.cross(normal, [1, 0, 0])
                        if np.linalg.norm(v1) < 1e-6:
                            v1 = np.cross(normal, [0, 1, 0])
                        v1 = v1 / np.linalg.norm(v1)
                        v2 = np.cross(normal, v1)
                        v2 = v2 / np.linalg.norm(v2)
                        # Build 2D coordinates in the plane
                        local_coords = np.dot(section_points - point, np.vstack([v1, v2]).T)
                        # Triangulate the 2D polygon
                        
                        tri = Delaunay(local_coords)
                        # Create mesh in 3D
                        plane_mesh = trimesh.Trimesh(vertices=section_points, faces=tri.simplices, process=False)

                # Split mesh by plane
                part = part.slice_plane(plane_origin=point, plane_normal=normal)

                part = trimesh.util.concatenate([part, plane_mesh])

                if verbose:
                    print(f"Cut {it+1}/{num_cuts} with plane at point {point} and normal {normal}")
                    print(part)
            if verbose:
                print(f"Fragment {it+1} created with {len(part.vertices)} vertices and {len(part.faces)} faces.")
            if part:
                # rot_matrix = rot_to_plane(normal)
                # Move the center of the fragment to (0, 0, 0)
                translation = -part.centroid
                part.apply_translation(translation)
                # Apply rotation to align with the camera view
                # plane_mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))
                fragments.append(part)

        except Exception as e:
            print(e)
    return fragments

def rot_to_plane(normal):
    # Target camera view direction (OpenGL: looking down -Z)
    camera_view = np.array([0, 0, -1])

    # Compute rotation to align plane_normal to camera_view
    rotation_axis = np.cross(normal, camera_view)
    if np.linalg.norm(rotation_axis) < 1e-8:
        # Already aligned or opposite; handle special case
        if np.dot(normal, camera_view) > 0:
            rot_matrix = np.eye(3)
        else:
            # 180 degree rotation around any perpendicular axis
            perp_axis = np.cross(normal, [1, 0, 0])
            if np.linalg.norm(perp_axis) < 1e-8:
                perp_axis = np.cross(normal, [0, 1, 0])
            perp_axis /= np.linalg.norm(perp_axis)
            rot_matrix = R.from_rotvec(np.pi * perp_axis).as_matrix()
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(normal, camera_view) /
                                  (np.linalg.norm(normal) * np.linalg.norm(camera_view)), -1.0, 1.0))
        rot_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()
    return rot_matrix

def render_fragment_views(fragment, out_dir, frag_idx): # not working, no intent to use
    # Set up 6 standard camera transforms
    views = {
        "front":  np.eye(4),
        "back":   trimesh.transformations.rotation_matrix(np.pi, [0,1,0]),
        "left":   trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0]),
        "right":  trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]),
        "top":    trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0]),
        "bottom": trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]),
    }
    scene = trimesh.Scene(fragment)
    for view_name, tf in views.items():
        # Set camera
        scene.camera.transform = tf
        # Render image
        png = scene.save_image(resolution=(512, 512), visible=True)
        if png is not None:
            img_path = os.path.join(out_dir, f"fragment_{frag_idx}_{view_name}.png")
            with open(img_path, 'wb') as f:
                f.write(png)

# Compute the best-fit plane for a set of points using PCA
def compute_best_fit_plane(frag, verbose=False): 
    points = frag.vertices
    if verbose:
        print(f"Fragment points:\n{points[:5]}...")

    centroid = frag.vertices.mean(axis=0)
    if verbose:
        print(f"Fragment centroid: {centroid}")

    # Compute covariance and eigenvectors (PCA)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if verbose:
        print(f"Fragment covariance:\n{cov}")
        print(f"Fragment eigenvalues: {eigvals}")
        print(f"Fragment eigenvectors:\n{eigvecs}")

    # The normal of the best-fit plane is the eigenvector with the smallest eigenvalue
    normal = eigvecs[:, np.argmin(eigvals)]

    return centroid, normal, eigvecs, eigvals

def determine_exterior_side(frag, normal, verbose=False):
    # Determine the side of the fragment that is closer to the convex hull
    # Compute centroid
    centroid = frag.vertices.mean(axis=0)
    # Signed distances
    signed_distances = np.dot(frag.vertices - centroid, normal)
    side1_mask = signed_distances >= 0
    side2_mask = signed_distances < 0

    # Faces whose all vertices are on one side
    faces = frag.faces
    side1_faces = np.where(np.all(side1_mask[faces], axis=1))[0]
    side2_faces = np.where(np.all(side2_mask[faces], axis=1))[0]

    # Create submeshes
    half1 = frag.submesh([side1_faces], append=True, repair=False)
    half2 = frag.submesh([side2_faces], append=True, repair=False)

    # Convex hulls
    hull1 = half1.convex_hull
    hull2 = half2.convex_hull
    convex_hull = frag.convex_hull

    vol_diff1 = abs(convex_hull.volume - hull1.volume)
    vol_diff2 = abs(convex_hull.volume - hull2.volume)

    # Choose camera direction (toward the fragment)
    if vol_diff1 < vol_diff2:
        if verbose:
            print('side1 is closer to convex')
    else:
        if verbose:
            print('side2 is closer to convex')

    return vol_diff1 < vol_diff2

def render_frag(frag, distance, volume, visualize=False):
    # Translate so centroid is at [0, 0, -distance]
    mean_position = frag.vertices.mean(axis=0)
    _volume = frag.volume if frag.volume > 0 else 1.0
    _distance = frag.extents.max() * 2
    scale = _distance / distance
    translation = -mean_position + np.array([0, 0, -_distance])
    frag.apply_translation(translation)
    # print('frag vertex colors', frag.visual.vertex_colors)
    # Prepare PyVista mesh
    faces = np.hstack([[3, *face] for face in frag.faces])  # PyVista expects [N, 3, v0, v1, v2, ...]
    pv_mesh = pv.PolyData(frag.vertices, faces)

    # Add vertex colors if available
    if hasattr(frag.visual, 'vertex_colors') and frag.visual.vertex_colors is not None:
        vcolors = frag.visual.vertex_colors
        if vcolors.shape[1] == 4:
            vcolors = vcolors[:, :3]
        pv_mesh.point_data['colors'] = (vcolors / 255.0)  # PyVista expects float in [0,1]
    # print('converted colors', pv_mesh.point_data['colors'])
    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.add_mesh(pv_mesh, 
                     scalars='colors' if 'colors' in pv_mesh.point_data else None, 
                     rgb=True,
                     smooth_shading=False,
                     ambient=0.5,
                     )
    plotter.set_background(color = [0.8, 0.8, 0.8], top = [0.85, 0.85, 0.85])
    plotter.camera_position = [(0, 0, 0), (0, 0, -_distance), (0, 1, 0)]
    plotter.camera.zoom(scale)

    # Render
    img = plotter.screenshot(return_img=True)
    plotter.close()

    if visualize:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return img

def render_frag_naive(frag, distance, volume, vertecies, visualize=False):
    # Translate so centroid is at [0, 0, -distance]
    mean_position = frag.vertices.mean(axis=0)
    _volume = frag.volume if frag.volume > 0 else 1.0
    _distance = frag.extents.max() * 2
    _vertices_count = len(frag.vertices)
    scale = _vertices_count / vertecies
    translation = -mean_position + np.array([0, 0, -_distance])
    frag.apply_translation(translation)
    # print('frag vertex colors', frag.visual.vertex_colors)
    # Prepare PyVista mesh
    faces = np.hstack([[3, *face] for face in frag.faces])  # PyVista expects [N, 3, v0, v1, v2, ...]
    pv_mesh = pv.PolyData(frag.vertices, faces)

    # Add vertex colors if available
    if hasattr(frag.visual, 'vertex_colors') and frag.visual.vertex_colors is not None:
        vcolors = frag.visual.vertex_colors
        if vcolors.shape[1] == 4:
            vcolors = vcolors[:, :3]
        pv_mesh.point_data['colors'] = (vcolors / 255.0)  # PyVista expects float in [0,1]
    # print('converted colors', pv_mesh.point_data['colors'])
    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.add_mesh(pv_mesh, 
                     scalars='colors' if 'colors' in pv_mesh.point_data else None, 
                     rgb=True,
                    #  pbr=True,
                     smooth_shading=True,
                     ambient=0.45,
                    #  metallic=0.5,
                    #  roughness=0.5,
                     diffuse=0.4,
                     specular=0.2,
                    #  lighting=True,
                     )
    plotter.set_background(color = [0.8, 0.8, 0.8], top = [0.85, 0.85, 0.85])
    plotter.camera_position = [(0, 0, 0), (0, 0, -_distance), (0, 1, 0)]

    plotter.camera.zoom(scale**(1/8))

    # Render
    img = plotter.screenshot(return_img=True)
    plotter.close()

    if visualize:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return img

def render_frag_realistic(frag, scale, visualize=False):
    # Translate so centroid is at [0, 0, -distance]
    mean_position = frag.vertices.mean(axis=0)
    _distance = frag.extents.max() * 2


    translation = -mean_position + np.array([0, 0, -_distance])
    frag.apply_translation(translation)
    # print('frag vertex colors', frag.visual.vertex_colors)
    # Prepare PyVista mesh
    faces = np.hstack([[3, *face] for face in frag.faces])  # PyVista expects [N, 3, v0, v1, v2, ...]
    pv_mesh = pv.PolyData(frag.vertices, faces)

    # Add vertex colors if available
    if hasattr(frag.visual, 'vertex_colors') and frag.visual.vertex_colors is not None:
        vcolors = frag.visual.vertex_colors
        if vcolors.shape[1] == 4:
            vcolors = vcolors[:, :3]
        pv_mesh.point_data['colors'] = (vcolors / 255.0)  # PyVista expects float in [0,1]
    # print('converted colors', pv_mesh.point_data['colors'])
    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.add_mesh(pv_mesh, 
                     scalars='colors' if 'colors' in pv_mesh.point_data else None, 
                     rgb=True,
                    #  pbr=True,
                     smooth_shading=True,
                     ambient=0.45,
                    #  metallic=0.5,
                    #  roughness=0.5,
                     diffuse=0.4,
                     specular=0.2,
                    #  lighting=True,
                     )
    plotter.set_background(color = [0.8, 0.8, 0.8], top = [0.85, 0.85, 0.85])
    plotter.camera_position = [(0, 0, 0), (0, 0, -_distance), (0, 1, 0)]

    plotter.camera.zoom(scale**(1/8))

    # Render
    img = plotter.screenshot(return_img=True)
    plotter.close()

    if visualize:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return img

def render_frag_pyrender(frag, visualize=False):
    # Visualize

    # Translate both so centroid is at [0, 0, -distance]
    mean_position = frag.vertices.mean(axis=0)
    distance = frag.extents.max() * 2
    translation = -mean_position + np.array([0, 0, -distance])
    frag.apply_translation(translation)
    
    # Set up pyrender scene
    scene = pyrender.Scene(ambient_light=[1, 1, 1])

    pyrender_mesh = pyrender.Mesh.from_trimesh(frag)
    scene.add(pyrender_mesh)

    # Camera at [0, 0, 0], looking at [0, 0, -1], up=[0, 1, 0]
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Render
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    if visualize:
        plt.imshow(color)
        plt.axis('off')
        plt.show()
    
    return color

def render_frag_material(frag, material, visualize=False):
    # Visualize

    # Translate both so centroid is at [0, 0, -distance]
    mean_position = frag.vertices.mean(axis=0)
    distance = frag.extents.max() * 2
    translation = -mean_position + np.array([0, 0, -distance])
    frag.apply_translation(translation)

    # Set up pyrender scene
    scene = pyrender.Scene()

    pyrender_mesh = pyrender.Mesh.from_trimesh(frag, material=material)
    scene.add(pyrender_mesh)

    # Camera at [0, 0, 0], looking at [0, 0, -1], up=[0, 1, 0]
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    if visualize:
        plt.imshow(color)
        plt.axis('off')
        plt.show()
    
    return color

def get_single_frag(config):
    mesh, verbose, visualize, save_frag, frag_savepath, save_render, render_savepath = config
    distance = mesh.extents.max() * 2
    volume = mesh.volume if mesh.volume > 0 else 1.0
    frag = None
    while frag is None:
        try:
            # t1 = time.time()
            frag = break_mesh(mesh, verbose=verbose)

            if frag is None:
                print("Failed to break mesh into fragments.")
                continue
            
            # frag = clean_trimesh(frag)
            frag = clean_mesh(frag, need_convert=True, verbose=verbose)
            # print('frag vertex colors', frag.visual.vertex_colors)
            

            mean_position, normal, eigvecs, eigvals = compute_best_fit_plane(frag, verbose=verbose)
            side1_is_closer = determine_exterior_side(frag, normal, verbose=verbose)

            rot_matrix = rot_to_plane(normal)
            rot180 = R.from_euler('y', 180, degrees=True).as_matrix() # 180 flip

            # Apply rotation to mesh (around mean_position)
            frag.apply_translation(-mean_position)
            frag.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))
            if not side1_is_closer:
                frag.apply_transform(np.vstack([np.hstack([rot180, np.zeros((3,1))]), [0,0,0,1]]))
            frag.apply_translation(mean_position)

            # Visualize
            color = render_frag(frag, distance, volume, visualize=visualize)
            # Save image
            if save_render and render_savepath is not None:
                folder_path = os.path.dirname(render_savepath)
                os.makedirs(folder_path, exist_ok=True)
                img = Image.fromarray(color)
                filename = os.path.basename(render_savepath)
                img.save(os.path.join(folder_path, f"rendered_{filename}.jpg"))
                if verbose:
                    print(f"Saved rendered image to {os.path.join(folder_path, f'rendered_{filename}.jpg')}")

            # Save fragments
            if save_frag and frag_savepath is not None:
                folder_path = os.path.dirname(frag_savepath)
                os.makedirs(folder_path, exist_ok=True)
                filename = os.path.basename(frag_savepath)
                frag.export(os.path.join(folder_path, f"fragment_{filename}.obj"))
                if verbose:
                    print(f"Saved fragment to {os.path.join(folder_path, f'fragment_{filename}.obj')}")
        except Exception as e:
            print(e)
            frag = None

    return frag

def run(obj_file, nb_cut=5, verbose=False, visualize=False):
    # Load your mesh
    mesh = trimesh.load(obj_file, force='mesh')  # Change to your mesh file
    for _ in range(nb_cut):
        frag = get_single_frag([mesh, verbose, visualize])
    return frag

def run_parallel(obj_file, nb_cut=5, threads=2, 
                 verbose=False, visualize=False,
                 save_frag=False, frag_save_path=None,
                 save_render=False, render_save_path=None):
    print(f"Running with {threads} threads...")
    print(f"Loading mesh from {obj_file}...")
    filename = os.path.basename(obj_file)[:-4]
    # Load your mesh
    mesh = trimesh.load(obj_file, force='mesh')  # Change to your mesh file
    meshes = [
        [
            mesh.copy(),
            verbose,
            visualize,
            save_frag,
            os.path.join(frag_save_path, filename, filename)+'_'+str(i+1) if frag_save_path else None,
            save_render,
            os.path.join(render_save_path, filename, filename)+'_'+str(i+1) if render_save_path else None,
        ]
        for i in range(nb_cut)
    ]
    
    executor = MultiThreadExecutor(get_single_frag)

    frags = []
    with ThreadPool(threads) as pool:
        for frag in tqdm(pool.imap(executor.opWrap, meshes), total=nb_cut):
            frags.append(frag)

    return frags

# Procrustes’ analysis to align fragment to model
def match_fragment_to_model(fragment, model):
    # Sample points from both meshes
    frag_points, _ = trimesh.sample.sample_surface(fragment, 10000)
    model_points, _ = trimesh.sample.sample_surface(model, 10000)

    # Procrustes alignment (rigid)
    matrix, transformation, cost = trimesh.registration.procrustes(frag_points, model_points, reflection=False, scale=False)
    fragment.apply_transform(matrix)

    # Transfer color: nearest neighbor from fragment vertices to model vertices
    if hasattr(model.visual, 'vertex_colors') and model.visual.vertex_colors is not None:
        model_tree = cKDTree(model.vertices)
        _, idx = model_tree.query(fragment.vertices)
        frag_colors = model.visual.vertex_colors[idx]
        fragment.visual.vertex_colors = frag_colors

    return fragment

def match_fragment_to_model_icp(fragment, model):
    # Sample points from both meshes
    frag_points, _ = trimesh.sample.sample_surface(fragment, 10000)
    model_points, _ = trimesh.sample.sample_surface(model, 10000)

    # Procrustes alignment (rigid)
    matrix, transformation, cost = trimesh.registration.icp(frag_points, model_points, max_iterations=40)
    fragment.apply_transform(matrix)

    # Transfer color: nearest neighbor from fragment vertices to model vertices
    if hasattr(model.visual, 'vertex_colors') and model.visual.vertex_colors is not None:
        model_tree = cKDTree(model.vertices)
        _, idx = model_tree.query(fragment.vertices)
        frag_colors = model.visual.vertex_colors[idx]
        fragment.visual.vertex_colors = frag_colors

    return fragment

def render_existing_mesh(obj_file, percentage, original_mesh, 
                         save_render = False, render_savepath = None, 
                         verbose=False, visualize=False):
    try:
        mesh = trimesh.load(obj_file, force='mesh')  # Change to your mesh file
        orig_mesh = trimesh.load(original_mesh, force='mesh')  # Change to your mesh file

        # colored_mesh = match_fragment_to_model(mesh, orig_mesh)
        colored_mesh = match_fragment_to_model_icp(mesh, orig_mesh)
        mesh = colored_mesh


        # rotate to best fit plane
        mean_position, normal, eigvecs, eigvals = compute_best_fit_plane(mesh, verbose=verbose)
        side1_is_closer = determine_exterior_side(mesh, normal, verbose=verbose)

        rot_matrix = rot_to_plane(normal)
        rot180 = R.from_euler('y', 180, degrees=True).as_matrix() # 180 flip

        # Apply rotation to mesh (around mean_position)
        mesh.apply_translation(-mean_position)
        mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))
        if not side1_is_closer:
            mesh.apply_transform(np.vstack([np.hstack([rot180, np.zeros((3,1))]), [0,0,0,1]]))
        mesh.apply_translation(mean_position)
        color = render_frag_realistic(mesh, percentage, visualize=visualize)
        # Save image
        if save_render and render_savepath is not None:
            os.makedirs(render_savepath, exist_ok=True)
            img = Image.fromarray(color)
            filename = os.path.basename(obj_file)[:-4]
            img.save(os.path.join(render_savepath, f"rendered_{filename}.jpg"))
            if verbose:
                print(f"Saved rendered image to {os.path.join(render_savepath, f'rendered_{filename}.jpg')}")
    except Exception as e:
        if verbose:
            print(f"Error rendering mesh {obj_file}: {e}")
        color = None
    return color

def render_existing_mesh_naive(obj_file, original_mesh, 
                               save_render = False, render_savepath = None, 
                               verbose=False, visualize=False):
    try:
        mesh = trimesh.load(obj_file, force='mesh')  # Change to your mesh file
        orig_mesh = trimesh.load(original_mesh, force='mesh')  # Change to your mesh file
        distance = orig_mesh.extents.max() * 2
        volume = orig_mesh.volume if mesh.volume > 0 else 1.0
        vertices_count = len(orig_mesh.vertices)
        
        # colored_mesh = match_fragment_to_model(mesh, orig_mesh)
        colored_mesh = match_fragment_to_model_icp(mesh, orig_mesh)
        mesh = colored_mesh


        # rotate to best fit plane
        mean_position, normal, eigvecs, eigvals = compute_best_fit_plane(mesh, verbose=verbose)
        side1_is_closer = determine_exterior_side(mesh, normal, verbose=verbose)

        rot_matrix = rot_to_plane(normal)
        rot180 = R.from_euler('y', 180, degrees=True).as_matrix() # 180 flip

        # Apply rotation to mesh (around mean_position)
        mesh.apply_translation(-mean_position)
        mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3,1))]), [0,0,0,1]]))
        if not side1_is_closer:
            mesh.apply_transform(np.vstack([np.hstack([rot180, np.zeros((3,1))]), [0,0,0,1]]))
        mesh.apply_translation(mean_position)
        color = render_frag_naive(mesh, distance, volume, vertices_count, visualize=visualize)
        # Save image
        if save_render and render_savepath is not None:
            os.makedirs(render_savepath, exist_ok=True)
            img = Image.fromarray(color)
            filename = os.path.basename(obj_file)[:-4]
            img.save(os.path.join(render_savepath, f"rendered_{filename}.jpg"))
            if verbose:
                print(f"Saved rendered image to {os.path.join(render_savepath, f'rendered_{filename}.jpg')}")
    except Exception as e:
        if verbose:
            print(f"Error rendering mesh {obj_file}: {e}")
        color = None
    return color

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(f"Argument: {arg}")
    if len(sys.argv) == 5:
        input_file = sys.argv[1]
        nb_cut = int(sys.argv[2])
        visualize = sys.argv[3].lower() in ['true', '1', 'yes']
        verbose = sys.argv[4].lower() in ['true', '1', 'yes']
        print("Starting...")
        run(input_file, nb_cut, verbose, visualize)
    if len(sys.argv) == 6:
        input_file = sys.argv[1]
        nb_cut = int(sys.argv[2])
        threads = int(sys.argv[3])
        visualize = sys.argv[4].lower() in ['true', '1', 'yes']
        verbose = sys.argv[5].lower() in ['true', '1', 'yes']
        print("Starting parallel...")
        run_parallel(input_file, nb_cut, threads, verbose, visualize)
    else:
        print("Usage: python rdm_3dbreak_trimesh.py <input_file> <nb_cut> <verbose> <visualize>")
        print("Example: python rdm_3dbreak_trimesh.py model.obj 5 True False")