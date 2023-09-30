import fiona
import re
import os
import pyproj
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr, gdalconst
from shapely.geometry import Point, Polygon, Point
from scipy.ndimage import binary_dilation

ogr.UseExceptions()
np.seterr(invalid="ignore")

# function: gtiff info
def tif_info(tif_path):
    dataset = gdal.Open(tif_path)

    if dataset is None:
        print("Failed to open the image.")
    else:
        print("Image opened successfully.")
        print(f"Number of bands: {dataset.RasterCount}")
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        print(f"Image dimensions: {width} x {height}")
        geotransform = dataset.GetGeoTransform()
        print("GeoTransform:", geotransform)
        projection = dataset.GetProjection()
        print("Projection:", projection)
        src_proj = pyproj.CRS.from_wkt(projection)
        epsg_code = src_proj.to_epsg()
        print("Projection epsg code:", epsg_code)

        metadata = dataset.GetMetadata()
        if metadata:
            print("metadata:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("can't find metadata")

        nodata_value = dataset.GetRasterBand(1).GetNoDataValue()
        print("NoData value:", nodata_value)

        print(gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType))

        dataset = None
        return projection


def value_count(array):
    unique_values, counts = np.unique(array, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")


# function: gpkg division
# processes the origan lake_polygons_training.gpkg and add layers spilted by image and region_num to the gpkg, totally 4 * 3 = 12 layers added
def extract_date_from_str(str):
    match = re.search(r"\d{4}-(\d{2})-(\d{2})", str)
    return (match.group(1), match.group(2)) if match else (None, None)


def spilt_training_gpkg(gpkg_path):
    layers = fiona.listlayers(gpkg_path)
    gdf = gpd.read_file(gpkg_path, layer=layers[0])
    print("training_gpkg layers:", layers)
    if len(layers) != 1:
        print("\ntraining_gpkg had been splitted")
    else:
        print("\nsplitting training_gpkg by image and region_num:")
        region_values = gdf["region_num"].unique()
        image_values = gdf["image"].unique()
        print(region_values)
        print(image_values)

        for region_value in region_values:
            for image_value in image_values:
                condition = (gdf["region_num"] == region_value) & (
                    gdf["image"] == image_value
                )
                sub_gdf = gdf[condition].copy()
                if not sub_gdf.empty:
                    month, day = extract_date_from_str(image_value)
                    if month != None and day != None:
                        layer_name = f"train_{month}_{day}_{region_value}"
                        sub_gdf.to_file(
                            gpkg_path, layer=layer_name, driver="GPKG", append=True
                        )
                        print(f"Saved {layer_name}")
        print("done")
        layers = fiona.listlayers(gpkg_path)
        print("training_gpkg layers after split:", layers)
    gdf = None


# function: raster division
# processes the origan raster to get sub-raster spilted by image and region.
def split_source_raster(gpkg_path, raster_path_list, output_folder):
    gpkg = ogr.Open(gpkg_path, 0)  # read only
    layer = gpkg.GetLayerByIndex(0)

    for raster_path in raster_path_list:
        raster = gdal.Open(raster_path)
        month, day = extract_date_from_str(raster_path)
        for feature in layer:
            output_tif_name = os.path.join(
                output_folder, f"{month}_{day}_{feature.GetFID()}.tif"
            )
            if os.path.exists(output_tif_name):
                print(f"{month}_{day}_{feature.GetFID()}.tif already exists.")
                continue
            gdal.Warp(
                output_tif_name,
                raster,
                cutlineDSName=gpkg_path,
                cropToCutline=True,
                cutlineWhere=f"FID={feature.GetFID()}",
                copyMetadata=True,
                format="GTiff",
                creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=1"],
                
            )
            print(f"{month}_{day}_{feature.GetFID()}.tif done.")
        raster = None
    gpkg.Release()


def cal_ndwi(raster_path, output_folder):
    output_path = os.path.join(output_folder, "ndwi_" + raster_path.rsplit("/", 1)[-1])
    if os.path.exists(output_path):
        print("ndwi_" + raster_path.rsplit("/", 1)[-1] + " already exists.")
        return

    rgb_tif = gdal.Open(raster_path, gdal.GA_ReadOnly)
    red_band_array = rgb_tif.GetRasterBand(1).ReadAsArray()
    blue_band_array = rgb_tif.GetRasterBand(3).ReadAsArray()
    ndwi_array = (1.0 * blue_band_array - red_band_array) / (
        1.0 * blue_band_array + red_band_array
    )
    driver = gdal.GetDriverByName("GTiff")
    ndwi_tif = driver.Create(
        output_path,
        rgb_tif.RasterXSize,
        rgb_tif.RasterYSize,
        1,
        gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=1"],
    )
    ndwi_tif.SetGeoTransform(rgb_tif.GetGeoTransform())
    ndwi_tif.SetProjection(rgb_tif.GetProjection())
    ndwi_tif.GetRasterBand(1).WriteArray(ndwi_array)
    ndwi_tif.GetRasterBand(1).SetNoDataValue(-9999)
    ndwi_tif.FlushCache()

    rgb_tif = None
    ndwi_tif = None
    print(
        raster_path.rsplit("/", 1)[-1] + " NDWI_ice calculation and saving completed."
    )


def get_arrays(ndwi_path, rgb_path, layer_num, gpkg_path = "data/lake_polygons_training.gpkg"):
    ndwi_dataset = gdal.Open(ndwi_path, gdal.GA_ReadOnly)
    rgb_dataset = gdal.Open(rgb_path, gdal.GA_ReadOnly)
    gpkg_dataset = gdal.OpenEx(gpkg_path, gdal.OF_VECTOR)
    ndwi_array = ndwi_dataset.GetRasterBand(1).ReadAsArray()  # -1 ~ 1, nan
    layer = gpkg_dataset.GetLayerByName("train_" + layer_num)
    if layer != None:
        lakes_dataset = gdal.GetDriverByName("MEM").Create(
            "", ndwi_dataset.RasterXSize, ndwi_dataset.RasterYSize, 1, gdal.GDT_Byte
        )
        lakes_dataset.SetGeoTransform(ndwi_dataset.GetGeoTransform())
        lakes_dataset.SetProjection(ndwi_dataset.GetProjection())
        lakes_band = lakes_dataset.GetRasterBand(1)
        lakes_band.SetNoDataValue(0)
        lakes_band.FlushCache()
        gdal.RasterizeLayer(
            lakes_dataset, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"]
        )
        lakes_array = lakes_band.ReadAsArray()  # 1, 0
    else:
        lakes_array = np.zeros_like(ndwi_array)

    tif_array = np.transpose(rgb_dataset.ReadAsArray(), (0, 1, 2))  # 0 255, 0

    lakes_dataset = None
    gpkg_dataset = None
    ndwi_dataset = None
    rgb_dataset = None

    return lakes_array, ndwi_array, tif_array


def create_sliced_data(lakes_array, tif_array, is_train_data=True):
    block_height = 256
    block_width = 256
    overlap = 0.5
    blocks_tifs = []
    blocks_masks = []
    _, height, width = tif_array.shape
    stride_height = int(block_height * (1 - overlap))
    stride_width = int(block_width * (1 - overlap))
    for y in range(0, height - stride_height, stride_height):
        if y + block_height > height:
            y = height - block_height
        for x in range(0, width - stride_width, stride_width):
            if x + block_width > width:
                x = width - block_width
            blocks_tif = tif_array[:, y : y + block_height, x : x + block_width]
            block_mask = lakes_array[y : y + block_height, x : x + block_width]
            if is_train_data is True:
                if not np.all(block_mask == 0):
                    blocks_tif = blocks_tif.transpose(1, 2, 0)
                    blocks_tifs.append(blocks_tif)
                    blocks_masks.append(block_mask)
            elif is_train_data is False:
                blocks_tif = blocks_tif.transpose(1, 2, 0)
                blocks_tifs.append(blocks_tif)
                blocks_masks.append(block_mask)
            else:
                return None, None
    blocks_tifs = np.array(blocks_tifs)
    blocks_masks = np.array(blocks_masks)
    return blocks_tifs, blocks_masks


def data_preprocess_train(
    train_layer_indices, val_layer_indices, layers, train_with_all_data=False
):
    train_tif, train_mask, val_tif, val_mask = [], [], [], []
    ndwi_dir = "data/ndwi ice/"
    rgb_dir = "data/splitted raster/"

    if(len(train_layer_indices) != 0):
        for index in train_layer_indices:
            photo_num = re.search(r"train_(.+)", layers[index]).group(1)
            ndwi_path = os.path.join(ndwi_dir, f"ndwi_{photo_num}.tif")
            rgb_path = os.path.join(rgb_dir, f"{photo_num}.tif")
            lakes_array, _, tif_array = get_arrays(
                ndwi_path, rgb_path, photo_num
            )
            sliced_tifs, sliced_masks = create_sliced_data(
                lakes_array, tif_array, is_train_data=not train_with_all_data
            )
            train_tif.append(sliced_tifs)
            train_mask.append(sliced_masks)
        train_tif = np.concatenate(train_tif, axis=0)
        train_mask = np.concatenate(train_mask, axis=0)

    if(len(val_layer_indices) != 0):
        for index in val_layer_indices:
            photo_num = re.search(r"train_(.+)", layers[index]).group(1)
            ndwi_path = os.path.join(ndwi_dir, f"ndwi_{photo_num}.tif")
            rgb_path = os.path.join(rgb_dir, f"{photo_num}.tif")
            lakes_array, _, tif_array = get_arrays(
                ndwi_path, rgb_path, photo_num
            )
            sliced_tifs, sliced_masks = create_sliced_data(
                lakes_array, tif_array, is_train_data=False
            )
            val_tif.append(sliced_tifs)
            val_mask.append(sliced_masks)
        val_tif = np.concatenate(val_tif, axis=0)
        val_mask = np.concatenate(val_mask, axis=0)

    return train_tif, train_mask, val_tif, val_mask, lakes_array, tif_array

def data_preprocess_test(
    layer
):
    test_tif, test_mask = [],[]
    ndwi_dir = "data/ndwi ice/"
    rgb_dir = "data/splitted raster/"

    photo_num = layer
    ndwi_path = os.path.join(ndwi_dir, f"ndwi_{photo_num}.tif")
    rgb_path = os.path.join(rgb_dir, f"{photo_num}.tif")
    lakes_array, ndwi_array, tif_array = get_arrays(
        ndwi_path, rgb_path, photo_num
    )
    sliced_tifs, sliced_masks = create_sliced_data(
        lakes_array, tif_array, is_train_data=False
    )
    test_tif.append(sliced_tifs)
    test_mask.append(sliced_masks)
    test_tif = np.concatenate(test_tif, axis=0)
    test_mask = np.concatenate(test_mask, axis=0)

    return test_tif, test_mask, tif_array, ndwi_array


def raster2vector(array, raster_path, vecter_path, field_name="class"):
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    lakes_dataset = gdal.GetDriverByName("MEM").Create("", array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    lakes_dataset.SetGeoTransform(raster.GetGeoTransform())
    lakes_dataset.SetProjection(raster.GetProjection())
    lakes_band = lakes_dataset.GetRasterBand(1)
    lakes_band.WriteArray(array)
    lakes_band.SetNoDataValue(0)
    lakes_band.FlushCache()

    band = lakes_dataset.GetRasterBand(1)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(vecter_path):
        drv.DeleteDataSource(vecter_path)

    polygon = drv.CreateDataSource(vecter_path)
    poly_layer = polygon.CreateLayer(
        vecter_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon
    )

    gdal.FPolygonize(band, band.GetMaskBand(), poly_layer, 0)

    polygon.SyncToDisk()
    polygon = None
    lakes_dataset = None
    raster = None


def cal_dem(tif_num):
    tif_path = "data/splitted raster/" + tif_num + ".tif"
    dem_paths = []
    gpkg_path = "data/lakes_regions.gpkg"
    gdf_dem_layers = gpd.read_file("data/lakes_regions.gpkg", layer = "dem_selected")
    gdf_regions = gpd.read_file(gpkg_path, layer = "regions")
    gdf_regions
    region = gdf_regions.iloc[[int(tif_num[-1]) - 1]]
    intersect = gpd.sjoin(gdf_dem_layers, region, how="inner", predicate="intersects")
    for index, row in intersect.iterrows():
        path = "data/dem raster/" + row['tile'] + "_dem.tif"
        dem_paths.append(path)

    tif_ds = gdal.Open(tif_path)
    tif_transform = tif_ds.GetGeoTransform()
    tif_proj = tif_ds.GetProjection()
    tif_array = tif_ds.GetRasterBand(1).ReadAsArray()
    dem_ds_list = []
    for dem_path in dem_paths:
        dem_ds = gdal.Open(dem_path)
        dem_ds = gdal.Warp('', dem_ds, format='MEM', dstSRS=tif_proj, resampleAlg=gdal.GRA_CubicSpline, xRes = tif_transform[1], yRes = tif_transform[1])
        dem_ds_list.append(dem_ds)

    options=gdal.WarpOptions(srcSRS=tif_proj, dstSRS=tif_proj,format='MEM',resampleAlg=gdalconst.GRA_CubicSpline)
    dem_ds = gdal.Warp("", dem_ds_list, options=options)

    dem_ds = gdal.Warp('', dem_ds, cutlineDSName=gpkg_path, cropToCutline=True,
        cutlineWhere=f"FID={int(tif_num[-1])}",
        copyMetadata=True,
        format="MEM",
    )
    slope_ds = gdal.DEMProcessing("", dem_ds, "slope", format="MEM", slopeFormat = "percent")
    slope_array = slope_ds.GetRasterBand(1).ReadAsArray()

    if slope_array.shape != tif_array.shape:
        slope_array = slope_array[:tif_array.shape[0], :tif_array.shape[1]]

    dem_ds = None
    tif_ds = None
    slope_ds = None
    return slope_array

def splice_image(origin_image, sliced_image_set):
    re_image = np.zeros_like(origin_image, dtype=np.uint8)
    block_height = 256
    block_width = 256
    overlap = 0.5
    stride_height = int(block_height * (1 - overlap))
    stride_width = int(block_width * (1 - overlap))
    height, width = origin_image.shape
    block_coordinates = []
    for y in range(0, height - stride_height, stride_height):
        if y + block_height > height:
            y = height - block_height
        for x in range(0, width - stride_width, stride_width):
            if x + block_width > width:
                x = width - block_width
            block_coordinates.append((y, x))
    for i, (block_mask, (y, x)) in enumerate(zip(sliced_image_set, block_coordinates)):
        re_image[y : y + block_height, x : x + block_width] |= block_mask
    return re_image

def cal_scores(y_true, y_pred):
    precision = (y_true & y_pred).sum() / y_pred.sum()
    recall = (y_true & y_pred).sum() / y_true.sum()
    f1 = 2 / (1 / precision + 1 / recall)
    print(f"p {precision:.4f}\tr {recall:.4f}\tf1 {f1:.4f}")
    return precision, recall, f1

def cal_scores_vector(gdf_true, gdf_pred):
    intersection = gpd.overlay(gdf_true, gdf_pred, how='intersection', keep_geom_type=False)
    tp = intersection.area.sum()
    fp = gdf_pred.area.sum() - tp
    fn = gdf_true.area.sum() - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / (1 / precision + 1 / recall)
    print(f"p {precision:.4f}\tr {recall:.4f}\tf1 {f1:.4f}")
    return precision, recall, f1

def array2gdf(array, tif_num):
    raster_path = "data/splitted raster/" + tif_num + ".tif"
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    lakes_dataset = gdal.GetDriverByName("MEM").Create("", array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    lakes_dataset.SetGeoTransform(raster.GetGeoTransform())
    lakes_dataset.SetProjection(raster.GetProjection())
    lakes_band = lakes_dataset.GetRasterBand(1)
    lakes_band.WriteArray(array)
    lakes_band.SetNoDataValue(0)
    lakes_band.FlushCache()
    band = lakes_dataset.GetRasterBand(1)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    drv = ogr.GetDriverByName("Memory")
    dataSource = drv.CreateDataSource('memData')
    layer = dataSource.CreateLayer('polygon', srs = prj, geom_type=ogr.wkbPolygon)
    gdal.FPolygonize(band, band.GetMaskBand(), layer, 0)
    features = []
    for feature in layer:
        features.append(feature.ExportToJson(as_object=True))
    gdf = gpd.GeoDataFrame.from_features(features, crs = 'EPSG:3857')
    lakes_dataset = None
    raster = None
    dataSource =None
    return gdf

def gdf2array(gdf, tif_num):
    raster_path = "data/splitted raster/" + tif_num + ".tif"
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    lakes_dataset = gdal.GetDriverByName("MEM").Create("", raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte)
    lakes_dataset.SetGeoTransform(raster.GetGeoTransform())
    lakes_dataset.SetProjection(raster.GetProjection())
    lakes_band = lakes_dataset.GetRasterBand(1)
    lakes_band.SetNoDataValue(0)
    lakes_band.FlushCache()
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')
    layer = mem_ds.CreateLayer("polygon", srs=prj, geom_type=ogr.wkbPolygon)
    field_defn = ogr.FieldDefn('geometry')
    layer.CreateField(field_defn)
    for index, row in gdf.iterrows():
        geom = ogr.CreateGeometryFromWkt(row['geometry'].wkt)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
        feature = None
    
    gdal.RasterizeLayer(
        lakes_dataset, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"]
    )
    lakes_array = lakes_band.ReadAsArray()

    lakes_dataset = None
    raster = None
    return lakes_array

def count_diameter_ratio(polygon):
    x, y = polygon.minimum_rotated_rectangle.exterior.xy
    a = Point(x[0], y[0]).distance(Point(x[1], y[1]))
    b = Point(x[2], y[2]).distance(Point(x[1], y[1]))
    longer_diameter = max(a, b)
    shorter_diameter = min(a, b)
    return longer_diameter / shorter_diameter


def post_process(gdf_pred, tif_num):
    # fill holes
    for index, row in gdf_pred.iterrows():
        geometry = row["geometry"]
        if len(geometry.interiors) > 0:
            gdf_pred.at[index, "geometry"] = Polygon(geometry.exterior)

    # delete area and diameter ratio
    gdf_pred["area"] = gdf_pred["geometry"].area
    gdf_pred = gdf_pred[gdf_pred["area"] >= 100000]
    gdf_pred = gdf_pred.drop(columns=["area"])
    
    true_array, ndwi_array, rgb_array = get_arrays(
        "data/ndwi ice/ndwi_" + tif_num + ".tif",
        "data/splitted raster/" + tif_num + ".tif",
        tif_num,
    )

    # ndwi
    pred_array = gdf2array(gdf_pred, tif_num)
    pred_array[(ndwi_array > 0.01) == 0] = 0

    # rgb
    rgb_mask_array = np.zeros_like(ndwi_array)
    rgb_mask_array[np.all((rgb_array < 100) & (rgb_array != 0), axis=0)] = 1

    structure_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    rgb_mask_array = binary_dilation(
        rgb_mask_array, structure=structure_element, iterations=10
    ).astype(int)
    gdf_rgb_mask = array2gdf(rgb_mask_array, tif_num)
    gdf_rgb_mask["area"] = gdf_rgb_mask["geometry"].area
    gdf_rgb_mask = gdf_rgb_mask[gdf_rgb_mask["area"] >= 100000000]
    gdf_rgb_mask = gdf_rgb_mask.drop(columns=["area"])

    for index, row in gdf_rgb_mask.iterrows():
        geometry = row["geometry"]
        if len(geometry.interiors) > 0:
            gdf_rgb_mask.at[index, "geometry"] = Polygon(geometry.exterior)

    rgb_mask_array = gdf2array(gdf_rgb_mask, tif_num)
    pred_array[rgb_mask_array == 1] = 0

    # slope
    slope_array = cal_dem(tif_num)
    pred_array[(slope_array > 5) == 1] = 0

    # fill holes
    gdf_pred = array2gdf(pred_array, tif_num)
    for index, row in gdf_pred.iterrows():
        geometry = row["geometry"]
        if len(geometry.interiors) > 0:
            gdf_pred.at[index, "geometry"] = Polygon(geometry.exterior)

    # delete area and diameter ratio
    gdf_pred["area"] = gdf_pred["geometry"].area
    gdf_pred = gdf_pred[gdf_pred["area"] >= 100000]
    gdf_pred = gdf_pred.drop(columns=["area"])
    gdf_pred["diameter_ratio"] = gdf_pred["geometry"].apply(lambda x: count_diameter_ratio(x))
    gdf_pred = gdf_pred[gdf_pred["diameter_ratio"] <= 10]
    gdf_pred = gdf_pred.drop(columns=["diameter_ratio"])

    ndwi_array, rgb_array, rgb_mask_array, true_array, pred_array, slope_array = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    del ndwi_array, rgb_array, rgb_mask_array, true_array, pred_array, slope_array
    return gdf_pred