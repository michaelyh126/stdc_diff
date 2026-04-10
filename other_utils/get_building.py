# import os
# import rasterio
# from rasterio.mask import mask
# import geopandas as gpd
# import numpy as np
#
#
# if __name__ == '__main__':
#     # --- 1. 设置输入和输出路径 ---
#     # 请将这里的路径替换为你自己的文件路径
#     image_path = 'D:\gaza\加沙/after.tif'  # 你的遥感影像路径
#     shapefile_path = 'D:\gaza\加沙/gaza_buildings.shp'  # 你的建筑矢量文件路径
#     output_dir = 'D:\gaza\\building_before'  # 输出文件夹的名称
#
#     # --- 2. 创建输出目录 ---
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"已创建输出目录: {output_dir}")
#
#     # --- 3. 打开并读取数据 ---
#     try:
#         # 使用 geopandas 读取 shapefile
#         buildings_gdf = gpd.read_file(shapefile_path)
#         # 使用 rasterio 读取影像
#         with rasterio.open(image_path) as src:
#             image_crs = src.crs
#             image_profile = src.profile
#
#             # --- 4. 检查坐标系是否匹配 ---
#             if buildings_gdf.crs != image_crs:
#                 print("警告：Shapefile 和影像的坐标系不匹配！")
#                 print(f"影像 CRS: {image_crs}")
#                 print(f"Shapefile CRS: {buildings_gdf.crs}")
#                 print("正在将 Shapefile 转换到影像的坐标系...")
#                 buildings_gdf = buildings_gdf.to_crs(image_crs)
#                 print("坐标系转换完成。")
#
#             # --- 5. 遍历每个建筑并进行裁剪 ---
#             total_buildings = len(buildings_gdf)
#             print(f"共找到 {total_buildings} 个建筑，开始处理...")
#
#             for index, building in buildings_gdf.iterrows():
#                 # 获取建筑的几何形状
#                 geoms = [building['geometry']]
#
#                 try:
#                     # 使用 rasterio.mask 进行裁剪
#                     # crop=True 会将输出图像的范围裁剪到几何形状的最小边界框
#                     out_image, out_transform = mask(src, geoms, crop=True)
#
#                     # --- (可选但推荐) 紧凑裁剪，去除多余的nodata边框 ---
#                     # mask的crop=True已经做了大部分工作，但可能仍有nodata行列
#                     # 我们可以进一步精确地裁剪掉这些行列
#                     # 找到所有不为nodata的像素
#                     # 注意：nodata值可能不同，src.nodata会获取到
#                     if src.nodata is not None:
#                         # 创建一个掩膜，True表示有效数据
#                         valid_data_mask = (out_image != src.nodata).any(axis=0)
#
#                         # 找到有效数据的行列范围
#                         rows = np.any(valid_data_mask, axis=1)
#                         cols = np.any(valid_data_mask, axis=0)
#
#                         # 获取裁剪后的范围
#                         y_min, y_max = np.where(rows)[0][[0, -1]]
#                         x_min, x_max = np.where(cols)[0][[0, -1]]
#
#                         # 裁剪数组和更新transform
#                         out_image = out_image[:, y_min:y_max + 1, x_min:x_max + 1]
#                         out_transform = src.window_transform(
#                             rasterio.windows.Window.from_slices(
#                                 (y_min, y_max + 1), (x_min, x_max + 1)
#                             )
#                         )
#
#                     # --- 6. 准备并保存结果 ---
#                     # 更新profile以适应新的裁剪图像
#                     out_profile = src.profile
#                     out_profile.update({
#                         "height": out_image.shape[1],
#                         "width": out_image.shape[2],
#                         "transform": out_transform
#                     })
#
#                     # 构建输出文件名
#                     # 假设你的shp文件有一个名为 'id' 或 'FID' 的唯一标识符字段
#                     # 如果没有，可以用index来命名
#                     building_id = building.get('id', index)  # 尝试获取'id'字段，没有则用index
#                     output_filename = os.path.join(output_dir, f"building_{building_id}.tif")
#
#                     # 写入新的tif文件
#                     with rasterio.open(output_filename, 'w', **out_profile) as dst:
#                         dst.write(out_image)
#
#                     print(f"({index + 1}/{total_buildings}) 已成功保存: {output_filename}")
#
#                 except Exception as e:
#                     print(f"处理建筑 {index} 时出错: {e}")
#
#     except FileNotFoundError as e:
#         print(f"错误：找不到文件 - {e}")
#     except Exception as e:
#         print(f"发生未知错误: {e}")
#
#     print("\n所有建筑处理完毕！")


# import os
# import rasterio
# from rasterio.mask import mask
# import geopandas as gpd
# import numpy as np
# from rasterio import features
#
# if __name__ == '__main__':
#     # --- 1. 设置输入和输出路径 ---
#     # 推荐使用原始字符串 r'...' 或正斜杠 '/' 来避免路径转义问题
#     image_path = r'D:\gaza\加沙\before.tif'  # 你的遥感影像路径
#     shapefile_path = r'D:\gaza\加沙\gaza_buildings.shp'  # 你的建筑矢量文件路径
#     output_dir = r'D:\gaza\building_new'  # 输出文件夹的名称 (修改为after以示区别)
#
#     # --- 2. 创建输出目录 ---
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"已创建输出目录: {output_dir}")
#
#     # --- 3. 打开并读取数据 ---
#     try:
#         # 使用 geopandas 读取 shapefile
#         buildings_gdf = gpd.read_file(shapefile_path)
#         # 使用 rasterio 读取影像
#         with rasterio.open(image_path) as src:
#             image_crs = src.crs
#             image_height = src.height
#             image_width = src.width
#
#             # --- 4. 检查坐标系是否匹配 ---
#             if buildings_gdf.crs != image_crs:
#                 print("警告：Shapefile 和影像的坐标系不匹配！")
#                 print(f"影像 CRS: {image_crs}")
#                 print(f"Shapefile CRS: {buildings_gdf.crs}")
#                 print("正在将 Shapefile 转换到影像的坐标系...")
#                 buildings_gdf = buildings_gdf.to_crs(image_crs)
#                 print("坐标系转换完成。")
#
#             # --- 5. 遍历每个建筑并进行裁剪 ---
#             total_buildings = len(buildings_gdf)
#             print(f"共找到 {total_buildings} 个建筑，开始处理...")
#
#             for index, building in buildings_gdf.iterrows():
#                 # 获取建筑的几何形状
#                 geoms = [building['geometry']]
#
#                 try:
#                     # 【新增】首先，获取建筑在原始影像中的大致像素范围，用于判断是否需要扩展
#                     # 这比先裁剪再判断更高效
#                     bounds = building.geometry.bounds
#                     # 将地理坐标转换为像素行列号
#                     # 注意：row对应y，col对应x
#                     row_start, col_start = src.index(bounds[0], bounds[3])  # minx, maxy
#                     row_end, col_end = src.index(bounds[2], bounds[1])  # maxx, miny
#
#                     # 计算建筑覆盖的像素尺寸（向上取整）
#                     pixel_height = int(np.ceil(row_end) - np.floor(row_start))
#                     pixel_width = int(np.ceil(col_end) - np.floor(col_start))
#
#                     # 【核心逻辑】判断尺寸并执行不同裁剪策略
#                     if pixel_height < 128 or pixel_width < 128:
#                         # --- 情况1: 需要扩展到128x128 ---
#                         print(
#                             f"({index + 1}/{total_buildings}) 建筑 {index} 尺寸 {pixel_width}x{pixel_height} < 128x128，正在扩展...")
#
#                         # 计算建筑中心在原始影像中的像素位置
#                         center_row = (row_start + row_end) / 2
#                         center_col = (col_start + col_end) / 2
#
#                         # 计算新的128x128窗口的边界，并进行边界检查
#                         new_row_start = max(0, int(np.floor(center_row - 64)))
#                         new_row_end = min(image_height, int(np.ceil(center_row + 64)))
#                         new_col_start = max(0, int(np.floor(center_col - 64)))
#                         new_col_end = min(image_width, int(np.ceil(center_col + 64)))
#
#                         # 创建新的窗口对象
#                         window = rasterio.windows.Window.from_slices(
#                             (new_row_start, new_row_end),
#                             (new_col_start, new_col_end)
#                         )
#
#                         window_data = src.read(window=window)
#                         # 获取该窗口的变换矩阵
#                         window_transform = src.window_transform(window)
#
#                         # 步骤2: 创建一个布尔掩膜，用于区分建筑物内外
#                         # invert=True 使得建筑物内部为True，外部为False
#                         # out_shape 需要与窗口数据的形状匹配 (height, width)
#                         geom_mask = features.geometry_mask(
#                             geoms,
#                             out_shape=window_data.shape[1:],
#                             transform=window_transform,
#                             invert=True
#                         )
#
#                         # 步骤3: 将掩膜应用到窗口数据上
#                         out_image = window_data.copy()
#                         if src.nodata is not None:
#                             # 将建筑物外的像素设置为nodata
#                             out_image[:, ~geom_mask] = src.nodata
#
#                         # 步骤4: 设置输出的变换矩阵
#                         out_transform = window_transform
#
#                     else:
#                         # --- 情况2: 尺寸足够，执行原有的紧凑裁剪 ---
#                         # 使用 rasterio.mask 进行裁剪
#                         out_image, out_transform = mask(src, geoms, crop=True)
#
#                         # --- (可选但推荐) 紧凑裁剪，去除多余的nodata边框 ---
#                         if src.nodata is not None:
#                             # 创建一个掩膜，True表示有效数据
#                             # .any(axis=0) 对所有波段进行操作，只要有一个波段不是nodata，就算有效数据
#                             valid_data_mask = (out_image != src.nodata).any(axis=0)
#
#                             # 找到有效数据的行列范围
#                             rows = np.any(valid_data_mask, axis=1)
#                             cols = np.any(valid_data_mask, axis=0)
#
#                             # 如果有效数据为空（例如，建筑在影像边缘外），则跳过
#                             if not np.any(rows) or not np.any(cols):
#                                 print(f"警告：建筑 {index} 在裁剪后无有效数据，已跳过。")
#                                 continue
#
#                             # 获取裁剪后的范围
#                             y_min, y_max = np.where(rows)[0][[0, -1]]
#                             x_min, x_max = np.where(cols)[0][[0, -1]]
#
#                             # 裁剪数组和更新transform
#                             out_image = out_image[:, y_min:y_max + 1, x_min:x_max + 1]
#                             out_transform = src.window_transform(
#                                 rasterio.windows.Window.from_slices(
#                                     (y_min, y_max + 1), (x_min, x_max + 1)
#                                 )
#                             )
#
#                     # --- 6. 准备并保存结果 ---
#                     # 更新profile以适应新的裁剪图像
#                     out_profile = src.profile
#                     out_profile.update({
#                         "height": out_image.shape[1],
#                         "width": out_image.shape[2],
#                         "transform": out_transform
#                     })
#
#                     # 构建输出文件名
#                     # 假设你的shp文件有一个名为 'id' 或 'osm_id' 的唯一标识符字段
#                     # 如果没有，可以用index来命名
#                     building_id = building.get('osm_id', building.get('id', index))  # 尝试获取常见ID字段
#                     output_filename = os.path.join(output_dir, f"building_{building_id}.tif")
#
#                     # 写入新的tif文件
#                     with rasterio.open(output_filename, 'w', **out_profile) as dst:
#                         dst.write(out_image)
#
#                     print(
#                         f"({index + 1}/{total_buildings}) 已成功保存: {output_filename} (尺寸: {out_image.shape[2]}x{out_image.shape[1]})")
#
#                 except Exception as e:
#                     print(f"处理建筑 {index} (ID: {building.get('osm_id', index)}) 时出错: {e}")
#
#     except FileNotFoundError as e:
#         print(f"错误：找不到文件 - {e}")
#     except Exception as e:
#         print(f"发生未知错误: {e}")
#
#     print("\n所有建筑处理完毕！")


import os
import rasterio
from rasterio.mask import mask
from rasterio import features  # 【新增】导入 features 模块
import geopandas as gpd
import numpy as np

if __name__ == '__main__':
    # --- 1. 设置输入和输出路径 ---
    # 推荐使用原始字符串 r'...' 或正斜杠 '/' 来避免路径转义问题
    image_path = r'D:\gaza\加沙\before.tif'  # 你的遥感影像路径
    shapefile_path = r'D:\gaza\加沙\gaza_buildings.shp'  # 你的建筑矢量文件路径
    output_dir = r'D:\gaza\building_new'  # 输出文件夹的名称

    # --- 2. 创建输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # --- 3. 打开并读取数据 ---
    try:
        # 使用 geopandas 读取 shapefile
        buildings_gdf = gpd.read_file(shapefile_path)
        # 使用 rasterio 读取影像
        with rasterio.open(image_path) as src:
            image_crs = src.crs
            image_height = src.height
            image_width = src.width

            # --- 4. 检查坐标系是否匹配 ---
            if buildings_gdf.crs != image_crs:
                print("警告：Shapefile 和影像的坐标系不匹配！")
                print(f"影像 CRS: {image_crs}")
                print(f"Shapefile CRS: {buildings_gdf.crs}")
                print("正在将 Shapefile 转换到影像的坐标系...")
                buildings_gdf = buildings_gdf.to_crs(image_crs)
                print("坐标系转换完成。")

            # --- 5. 遍历每个建筑并进行裁剪 ---
            total_buildings = len(buildings_gdf)
            print(f"共找到 {total_buildings} 个建筑，开始处理...")

            for index, building in buildings_gdf.iterrows():
                # 获取建筑的几何形状
                geoms = [building['geometry']]

                try:
                    # 【新增】首先，获取建筑在原始影像中的大致像素范围，用于判断是否需要扩展
                    bounds = building.geometry.bounds
                    row_start, col_start = src.index(bounds[0], bounds[3])
                    row_end, col_end = src.index(bounds[2], bounds[1])

                    pixel_height = int(np.ceil(row_end) - np.floor(row_start))
                    pixel_width = int(np.ceil(col_end) - np.floor(col_start))

                    # 【核心逻辑】判断尺寸并执行不同裁剪策略
                    if pixel_height < 64 or pixel_width < 64:
                        # --- 情况1: 需要扩展到128x128 ---
                        print(
                            f"({index + 1}/{total_buildings}) 建筑 {index} 尺寸 {pixel_width}x{pixel_height} < 128x128，正在扩展...")

                        # 计算建筑中心在原始影像中的像素位置
                        center_row = (row_start + row_end) / 2
                        center_col = (col_start + col_end) / 2

                        # 计算新的128x128窗口的边界，并进行边界检查
                        new_row_start = max(0, int(np.floor(center_row - 32)))
                        new_row_end = min(image_height, int(np.ceil(center_row + 32)))
                        new_col_start = max(0, int(np.floor(center_col - 32)))
                        new_col_end = min(image_width, int(np.ceil(center_col + 32)))

                        # 创建新的窗口对象
                        window = rasterio.windows.Window.from_slices(
                            (new_row_start, new_row_end),
                            (new_col_start, new_col_end)
                        )

                        # 【修改后的正确逻辑】
                        # 步骤1: 从源影像中读取指定窗口的数据
                        window_data = src.read(window=window)
                        # 获取该窗口的变换矩阵
                        window_transform = src.window_transform(window)

                        # 步骤2: 创建一个布尔掩膜，用于区分建筑物内外
                        # invert=True 使得建筑物内部为True，外部为False
                        # out_shape 需要与窗口数据的形状匹配 (height, width)
                        geom_mask = features.geometry_mask(
                            geoms,
                            out_shape=window_data.shape[1:],
                            transform=window_transform,
                            invert=True
                        )

                        # 步骤3: 将掩膜应用到窗口数据上
                        out_image = window_data.copy()
                        if src.nodata is not None:
                            # 将建筑物外的像素设置为nodata
                            out_image[:, ~geom_mask] = src.nodata

                        # 步骤4: 设置输出的变换矩阵
                        out_transform = window_transform

                    else:
                        # --- 情况2: 尺寸足够，执行原有的紧凑裁剪 ---
                        out_image, out_transform = mask(src, geoms, crop=True)

                        # --- (可选但推荐) 紧凑裁剪，去除多余的nodata边框 ---
                        if src.nodata is not None:
                            valid_data_mask = (out_image != src.nodata).any(axis=0)
                            rows = np.any(valid_data_mask, axis=1)
                            cols = np.any(valid_data_mask, axis=0)

                            if not np.any(rows) or not np.any(cols):
                                print(f"警告：建筑 {index} 在裁剪后无有效数据，已跳过。")
                                continue

                            y_min, y_max = np.where(rows)[0][[0, -1]]
                            x_min, x_max = np.where(cols)[0][[0, -1]]

                            out_image = out_image[:, y_min:y_max + 1, x_min:x_max + 1]
                            out_transform = src.window_transform(
                                rasterio.windows.Window.from_slices(
                                    (y_min, y_max + 1), (x_min, x_max + 1)
                                )
                            )

                    # --- 6. 准备并保存结果 ---
                    out_profile = src.profile
                    out_profile.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    building_id = building.get('osm_id', building.get('id', index))
                    output_filename = os.path.join(output_dir, f"building_{building_id}.tif")

                    with rasterio.open(output_filename, 'w', **out_profile) as dst:
                        dst.write(out_image)

                    print(
                        f"({index + 1}/{total_buildings}) 已成功保存: {output_filename} (尺寸: {out_image.shape[2]}x{out_image.shape[1]})")

                except Exception as e:
                    print(f"处理建筑 {index} (ID: {building.get('osm_id', index)}) 时出错: {e}")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print("\n所有建筑处理完毕！")
