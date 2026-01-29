import open3d as o3d
import numpy as np
from numpy.ma.core import floor
from caulate_depth import estimate_depth_for_non_uniform
from filter_mesh_by_proximity_density import filter_mesh_cascaded
from filter_mesh_by_proximity import filter_mesh_by_proximity
from iterative_boundary_smoothing import iterative_boundary_smoothing
from Screened_poisson import Screened_poisson
from MLS_filter import mls_smoothing_knn
from cascade_filetr import filter_mesh_cascaded

if __name__ == "__main__":
    pcd_path = r"../data/CISI/leaf_CISI9.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)

    # downsample+SOR+MLS
    SOR_pcd, ind= pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
    SOR_points = np.asarray(SOR_pcd.points)
    # pcd_points = np.asarray(pcd.points)
    smooth_pcd = mls_smoothing_knn(SOR_points, 15, 2)
    voxel_size = 0.004
    sample_pcd = smooth_pcd.voxel_down_sample(voxel_size)


    if len(layer_pcd.points) >= 10:
        depth = estimate_depth_for_non_uniform(sample_pcd, 0.01, target_points_per_leaf=1)
        point_weight = 128
        radius = voxel_size
        projected_pcd, mesh, hit_mesh, density= Screened_poisson(smooth_pcd, sample_pcd, depth , point_weight, radius)
        mesh.compute_vertex_normals()
        
        distances = projected_pcd.compute_nearest_neighbor_distance()
        PROXIMITY_THRESHOLD = np.max(distances) / np.sqrt(2)
        vertex_threshold = np.max(distances)

        filtered_mesh_combined = filter_mesh_cascaded(
            mesh,
            projected_pcd,
            centorid_threshold = PROXIMITY_THRESHOLD,
            vertex_threshold = vertex_threshold,
            depth = depth,
            radius = radius
        )
        
        ITERATIONS = 10
        LAMBDA_VAL = 0.2

        smoothed_mesh = iterative_boundary_smoothing(
            filtered_mesh_combined,
            iterations=ITERATIONS,
            lambda_val=LAMBDA_VAL
        )


        
    total_area = smoothed_mesh.get_surface_area()
    print(f"total_area: {total_area:.8f}")




