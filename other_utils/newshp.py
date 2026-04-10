if __name__ == '__main__':
    import geopandas as gpd

    # 1. 读取原始 SHP 文件（替换为你的文件路径）
    gdf = gpd.read_file("D:/gaza/加沙/gaza_buildings.shp")  # 例如："D:/data/original.shp"

    # 2. 筛选 fclass 字段值为 "ruins" 的记录
    # 🔍 重要：请先检查字段名是否正确（可通过 print(gdf.columns) 验证）
    ruins_gdf = gdf[gdf["fclass"] == "ruins"]

    # 3. 保存为新 SHP 文件（驱动器指定为 ESRI Shapefile）
    ruins_gdf.to_file("D:/gaza/ruins_output.shp", driver="ESRI Shapefile")
