from lib_array import *
from lib_skeleton3d import Skeleton
import matplotlib.pyplot as plt
import numpy as np
import shutil
import copy

def fit_plane_first_last(skeleton: np.ndarray):
    middle_point = skeleton[len(skeleton)//2]
    first_point = skeleton[0]
    last_point = skeleton[-1]
    
    vector1 = first_point - middle_point
    vector2 = last_point - middle_point
    normal = np.cross(vector1, vector2)
    return normal / np.linalg.norm(normal)

def get_flow_vector(skeleton: np.ndarray):
    middle_point_id = len(skeleton)//2
    flow_vector = skeleton[middle_point_id+1] - skeleton[middle_point_id-1]
    return flow_vector / np.linalg.norm(flow_vector)

def project_vector(n: np.ndarray, v: np.ndarray):
    projected = v - (np.dot(v, n) / np.dot(n, n) * n)
    return projected / np.linalg.norm(projected)

def rotate_vector(f: np.ndarray, n: np.ndarray, degrees: float):
    rad_angle = np.radians(degrees)
    
    add1 = n * np.cos(rad_angle)
    add2 = np.cross(f, n) * np.sin(rad_angle)
    add3 = np.dot(f, n) * (1 - np.cos(rad_angle)) * f
    return add1 + add2 + add3

def calculate_view_angle(plane_normal):
    normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    azimuth = np.degrees(np.arctan2(normal[1], normal[0]))
    elevation = np.degrees(np.arcsin(normal[2]))
    return elevation, azimuth

def project_to_2d(points, normal):
    normal = normal / np.linalg.norm(normal)
    u = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0, 1, 0])
    v = np.cross(normal, u)
    
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return np.array([[np.dot(p, u), np.dot(p, v)] for p in points])

def calc_rot_matrix(points, custom_angle: int = 0):
    first_point, last_point = points[0], points[-1]
    angle = np.arctan2(last_point[1] - first_point[1], last_point[0] - first_point[0])
    if custom_angle != 0:
        angle += custom_angle * np.pi/180
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    return rotation_matrix
    
def apply_rotation(points, matrix: np.ndarray, center = None) -> np.ndarray:
    if center is None:
        center = points[0]
    return (points - center) @ matrix.T

def get_values(image: np.ndarray, coords: np.ndarray):
    x_size, y_size, z_size = image.shape
    coords = np.clip(coords, [0, 0, 0], [x_size-1, y_size-1, z_size-1]).astype(int)
    result = image[coords[:, 0], coords[:, 1], coords[:, 2]]
    return result

def filter_branch(annotations: np.ndarray, skeleton: Skeleton):
    flooded = floodfill_3d_mask(annotations, skeleton[0].coordinates)
    isolated_skeleton = np.zeros_like(flooded).astype(bool)

    for point in skeleton.to_numpy():
        isolated_skeleton[tuple(point)] = True

    isolated_skeleton = dilate_3d(isolated_skeleton, kernel_size=3, iterations=10)
    combined_isolation = np.logical_and(flooded, isolated_skeleton)
    return np.argwhere(combined_isolation)

def recreate_output_folder(output_folder: str, branches: list[Skeleton]):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.mkdir(output_folder)
    for branch_id, _ in enumerate(branches):
        branch_specific_output_folder = f'{output_folder}/branch{branch_id}'
        if os.path.exists(branch_specific_output_folder):
            shutil.rmtree(branch_specific_output_folder)
        os.mkdir(branch_specific_output_folder)

def filter_projected_points(point_array: np.ndarray, color_array: np.ndarray, scale: float = 1.0):
    pixels = np.round(point_array * scale).astype(int)
    unique = {}
    for i, p in enumerate(pixels):
        key = tuple(p)
        unique[key] = i
    final_indices = list(unique.values())
    points_final = point_array[final_indices]
    colors_final = color_array[final_indices]
    return points_final, colors_final

def create_mpr(colored_image: np.ndarray,
               calcinate_mask: np.ndarray,
               branches: list[Skeleton],
               output_folder: str,
               rotations: int,
               colored: bool):

    if colored:
        colored_image = np.where(calcinate_mask, 1000, colored_image)

    recreate_output_folder(output_folder, branches)
    rotation_degrees = 360//rotations

    for branch_id, skeleton in enumerate(branches):
        original_skeleton = np.array([b.coordinates for b in skeleton])
        middle_point = original_skeleton[len(original_skeleton)//2]
        skeleton = original_skeleton - middle_point

        plane_normal = fit_plane_first_last(skeleton)
        flow_vector = project_vector(plane_normal, get_flow_vector(skeleton))

        plt.style.use('grayscale')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        for i in tqdm(range(rotations), ncols=100, desc=f'branch{branch_id}'):
            ax.cla()
            ax.set_aspect('equal')
            ax.axis('off')
            fig.patch.set_facecolor('black')

            rotated_plane = rotate_vector(flow_vector, plane_normal, i*rotation_degrees)
            
            first_to_last_vector = original_skeleton[-1] - original_skeleton[0]
            perpendicular_vector = np.cross(first_to_last_vector, rotated_plane)
            perpendicular_vector /= np.linalg.norm(perpendicular_vector)

            steps = np.arange(-200, 201, 1)
            steps = np.array([(s*perpendicular_vector) for s in steps])

            point_array = []
            color_array = []
            for point in original_skeleton:
                point_coords = steps + point
                point_values = get_values(colored_image, steps + point).astype(int)
                projected_point_coords = project_to_2d(point_coords, rotated_plane)
                point_array.extend(projected_point_coords)
                color_array.extend(point_values)
            color_array = np.array(color_array)

            if colored:
                gray = np.clip(color_array, 0, 255) / 255.0
                colors = np.stack([gray, gray, gray], axis=1)
                colors[color_array == 1000] = [1.0, 0.0, 0.0]
                color_array = colors
                

            projected_original_skeleton = project_to_2d(original_skeleton, rotated_plane)
            original_pivot = copy.deepcopy(projected_original_skeleton[0])
            rotation_matrix = calc_rot_matrix(projected_original_skeleton - original_pivot, custom_angle=90)
            point_array = apply_rotation(point_array, rotation_matrix, center=original_pivot)
            
            point_array, color_array = filter_projected_points(point_array, color_array, scale=0.8)
            ax.scatter(*zip(*point_array), c=color_array)
            
            branch_name = f'{output_folder}/branch{branch_id}'
            plt.savefig(f'{branch_name}/mpr{i*rotation_degrees}deg.png', dpi=100, facecolor='black')