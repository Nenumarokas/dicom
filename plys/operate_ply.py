import open3d as o3d
import numpy as np
import trimesh
import os

print('\n \n ')

def display(pointcloud):
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    pointcloud_name = 'final.ply'
    pcd = o3d.io.read_point_cloud(f'{os.getcwd()}\\{pointcloud_name}')
    
    print(1)
    pcd.estimate_normals()
    print(2)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   
    print(3)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))

    print(4)
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                            vertex_normals=np.asarray(mesh.vertex_normals))
    print(5)
    trimesh.convex.is_convex(tri_mesh)
    