import open3d as o3d
import numpy as np
from cc_downsample import cc_spatial_subsample
from numpy.ma.core import floor
from slice_point_cloud import slice_point_cloud
from caulate_depth import estimate_depth_for_non_uniform
from filter_mesh_by_proximity_density import filter_mesh_cascaded
from filter_mesh_by_proximity import filter_mesh_by_proximity
from iterative_boundary_smoothing import iterative_boundary_smoothing
from Screened_poisson import Screened_poisson
from voxelize import group_points_by_voxel
from MLS_filter import mls_smoothing_knn
from cascade_filetr import filter_mesh_cascaded
from edge_detect import edge_detect
from octree_minnode_length import octree_minnode_length

if __name__ == "__main__":
    # pcd_path = "../data/Leaf/leaf_5.pcd"
    # pcd_path = "../data/yulan_2.pcd"
    # pcd_path = "../data/CISI_1mm/leaf_CISI1_real.pcd"
    # pcd_path = "../data/CISI_4mm/leaf_CISI1.pcd"
    # pcd_path = r"../data/CISI/poisson_dis.pcd"
    # pcd_path = "../data/leaf_3-2.pcd"
    # pcd_path = r"../data/Leaf/shiceyepian2/leaf_15.pcd"
    # pcd_path = r"../data/highbrand/two_leaf.pcd"
    # pcd_path = r"../data/figure_leaf/leaf_3.pcd"
    pcd_path = r"../data/framegraph/leaf_CISI9.pcd"
    # pcd_path = r"../data/Wenjiangdataset/G3/8_SOR.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)


    # 点云预处理downsample+SOR+MLS
    SOR_pcd, ind= pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
    SOR_points = np.asarray(SOR_pcd.points)
    # pcd_points = np.asarray(pcd.points)
    smooth_pcd = mls_smoothing_knn(SOR_points, 15, 2)
    voxel_size = 0.004#采样距离
    sample_pcd = cc_spatial_subsample(smooth_pcd, voxel_size)
    o3d.io.write_point_cloud("../result/framework/framework.pcd", sample_pcd)

    # print("-------对输入点云体素化分-----------")
    layer_thickness = 10
    layers = slice_point_cloud(sample_pcd, layer_height=layer_thickness, axis=2)  # 按Z轴分层
    # VOXEL_SIZE = 10
    # layers = group_points_by_voxel(smooth_pcd, voxel_size=VOXEL_SIZE)


    # 用于存放每层的有效网格
    list_of_valid_meshes = []
    projected_final = o3d.geometry.PointCloud()


    for i, layer_pcd in enumerate(layers):
        if len(layer_pcd.points) >= 10:
            print(f"--- 正在处理第 {i} 层 ---")
            print("-----进行筛选泊松重建-------")
            depth = estimate_depth_for_non_uniform(sample_pcd, 0.01, target_points_per_leaf=1)
            # if depth < 8:
            #     depth = 8
            point_weight = 128
            radius = voxel_size
            projected_pcd, mesh, hit_mesh, density= Screened_poisson(smooth_pcd, layer_pcd, 5 , point_weight, radius)
            o3d.io.write_point_cloud("../result/framework/projected.pcd", projected_pcd)

            projected_final += projected_pcd

            mesh.compute_vertex_normals()
            output_filename_finalmesh = "../result/framework/origin_leaf_mesh.ply"  # .ply 和 .obj 是常用的格式
            o3d.io.write_triangle_mesh(output_filename_finalmesh, mesh)

            print("-----有效三角面片筛选-------")
            distances = projected_pcd.compute_nearest_neighbor_distance()
            PROXIMITY_THRESHOLD = np.max(distances) / np.sqrt(2)
            vertex_threshold = np.max(distances)
            print(f"------正在以阈值{PROXIMITY_THRESHOLD:.8f}进行筛选")
            # PROXIMITY_THRESHOLD = 0.001

            # filtered_mesh_combined = filter_mesh_cascaded(
            #     mesh,
            #     density,
            #     projected_pcd,
            #     PROXIMITY_THRESHOLD,
            #     radius
            # )

            filtered_mesh_combined = filter_mesh_cascaded(
                mesh,
                projected_pcd,
                centorid_threshold = PROXIMITY_THRESHOLD,
                vertex_threshold = vertex_threshold,
                depth = 8,
                radius = radius
            )


            # L_node, diag= octree_minnode_length(layer_pcd, depth)
            # print(f"max_iterations：{L_node}:.8f")
            # # peel_stop_threshold = L_node * np.sqrt(3) / 2
            #
            # max_iterations = np.int32(np.ceil(PROXIMITY_THRESHOLD / (radius * np.sqrt(3))))
            # print(f"max_iterations：{max_iterations}:.8f")
            # filtered_mesh_combined = edge_detect(mesh, projected_pcd , PROXIMITY_THRESHOLD, 0)

            # filtered_mesh_combined = filter_mesh_by_proximity(mesh, projected_pcd, PROXIMITY_THRESHOLD)


            #-----可视化筛选结果-------
            # filtered_mesh_combined.compute_vertex_normals()
            # projected_pcd.paint_uniform_color([0, 0, 1])  # 点云为蓝色
            # layer_pcd.paint_uniform_color([1, 0, 0])  # 点云为红色
            # o3d.visualization.draw_geometries(
            #     [filtered_mesh_combined, projected_pcd, layer_pcd],
            #     window_name="拼接后的最终三角网格",
            #     mesh_show_wireframe=True  # 同时显示线框，看得更清楚
            # )


            print("------进行迭代式边界平滑-----")
            # smoothed_mesh = iterative_shrink_wrap(filtered_mesh_enhanced, projected_pcd, 20)

            ITERATIONS = 10
            LAMBDA_VAL = 0.2

            smoothed_mesh = iterative_boundary_smoothing(
                filtered_mesh_combined,
                iterations=ITERATIONS,
                lambda_val=LAMBDA_VAL
            )

            # print("-------计算最终三角网的面积------")
            # total_area += smoothed_mesh.get_surface_area()

            # 将这个有效的网格添加到列表中
            list_of_valid_meshes.append(smoothed_mesh)
            area = smoothed_mesh.get_surface_area()
            print(f"自下而上这一层的面积：{area:.8f}")
        else:
            print("    -> 点数过少，跳过泊松重建。")
            continue
    print("-------存储投影后的点云为："+"posong_projected_1.pcd-----------")
    # o3d.io.write_point_cloud("../result/framework/projected.pcd", projected_final)

    # ---------------将列表中的所有网格对象合并成一个------------------
    final_mesh = o3d.geometry.TriangleMesh()# 从一个空网格开始，逐个相加
    for mesh in list_of_valid_meshes:
        final_mesh += mesh

    print("-------计算最终三角网的面积------")
    total_area = final_mesh.get_surface_area()
    print(f"总面积: {total_area:.8f}")


    print("--------可视化最终结果----------")
    final_mesh.compute_vertex_normals()# 为最终的网格计算法线，以获得更好的渲染效果
    projected_final.paint_uniform_color([0, 0, 1])  # 投影点云为蓝色
    final_mesh.paint_uniform_color([1, 0, 0])  # 原始点云为红色

    o3d.visualization.draw_geometries(
        [final_mesh, projected_final, smooth_pcd],
        window_name="拼接后的最终三角网格",
        mesh_show_wireframe=True,  # 同时显示线框，看得更清楚
        mesh_show_back_face = True
    )

    print("-------存储最终的叶片："+"combined_mesh.ply-----------")

    #
    # mesh.compute_vertex_normals()
    # output_filename_finalmesh = "../result/mesh_CISI1.ply"  # .ply 和 .obj 是常用的格式
    # o3d.io.write_triangle_mesh(output_filename_finalmesh, mesh)

    output_filename_finalmesh = "../result/framework/smoothed_leaf_mesh.ply"  # .ply 和 .obj 是常用的格式
    o3d.io.write_triangle_mesh(output_filename_finalmesh, final_mesh)
    print(f"\n已将拼接后的网格导出到文件: {output_filename_finalmesh}")


