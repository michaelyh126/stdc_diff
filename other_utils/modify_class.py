import os
import re
import geopandas as gpd
if __name__ == '__main__':

    # ------------------------------
    # 配置参数（需根据实际情况修改）
    shp_file_path = "D:\gaza\加沙\gaza_buildings.shp"  # 独立shp文件的绝对路径
    tif_folder_path = "D:\gaza\\buildings"  # 存放tif文件的文件夹路径
    # ------------------------------
    # 1. 读取独立的shp文件
    try:
        gdf = gpd.read_file(shp_file_path)
    except Exception as e:
        print(f"[错误] 无法读取shp文件: {shp_file_path}\n原因: {str(e)}")
        exit()
    # 2. 验证shp文件的关键字段
    required_cols = ["osm_id", "fclass"]
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        print(f"[错误] shp文件缺少必要字段: {missing_cols}")
        exit()
    # 3. 遍历tif文件夹，处理每个tif文件
    modified_count = 0  # 统计成功修改的记录数
    for file_name in os.listdir(tif_folder_path):
        if file_name.endswith(".tif"):
            # 提取tif文件名中的osm_id（匹配 "building_123456789.tif" 格式）
            match = re.match(r"^building_(\d+)\.tif$", file_name)
            if not match:
                print(f"[警告] tif文件名格式不符，跳过: {file_name}")
                continue
            osm_id = int(match.group(1))  # 提取osm_id（如1000805552）
            # 在shp中查找osm_id匹配的记录
            target_mask = gdf["osm_id"] == str(osm_id)
            if target_mask.any():
                # 修改fclass字段为'ruins'
                gdf.loc[target_mask, "fclass"] = "buildings"
                modified_count += 1
                print(f"[成功] 已修改shp中osm_id={osm_id}的fclass为ruins")
            else:
                print(f"[警告] shp中无osm_id={osm_id}的记录，跳过")
    # 4. 保存修改后的shp文件（覆盖原文件）
    try:
        gdf.to_file(shp_file_path, driver="ESRI Shapefile")
        print(f"\n[完成] 共修改{modified_count}条记录，shp文件已保存至: {shp_file_path}")
    except Exception as e:
        print(f"[错误] 无法保存shp文件: {shp_file_path}\n原因: {str(e)}")
