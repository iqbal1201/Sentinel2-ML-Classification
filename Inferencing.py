# IMPORT LIBRARY

import time
from datetime import date, datetime as dt
print('[START INFERENCING IMAGERY SCRIPT : {}] \n'.format(dt.now()).center(50, ' '))
init_processing_start_time = time.time()

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LogisticRegression
from re import search

from zipfile import ZipFile
import os, shutil
import pickle

success_import = False

while not success_import:
    try:
        import arcpy
        from arcgis.gis import GIS
        from arcgis.raster import ImageryLayer
        from arcgis.geometry import Geometry
        from arcgis.geometry.filters import intersects
        from arcgis.features import GeoAccessor
        from arcgis.features.analysis import join_features
        success_import = True
    except:
        pass

pd.options.mode.chained_assignment = None     
arcpy.env.overwriteOutput = True
arcpy.env.preserveGlobalIds = True
arcpy.SetLogHistory(False)

#CONFIGURABLE PARAMETERS
# Set Default Environment

# Environment Geodatabase
arcpy.env.workspace = r'\\10.16.0.10\sentinel\deforestation\intermediate_process\dev_gdb.gdb'

# Environment Crawling
crawling_path = r'\\10.16.0.10\sentinel\deforestation\intermediate_process\crawling'

# Environment Inferencing
infer_path = r'\\10.16.0.10\sentinel\deforestation\intermediate_process\inferencing'

inference_gdb = os.path.join(infer_path, "inference_temp.gdb")
extracted_polygon_temp = inference_gdb + "\\extracted_polygon_temp"
extracted_polygon_temp_filtered = inference_gdb + "\\extracted_polygon_temp_filtered"

#FUNCTION STATEMENT
def delete_files(folder_path):
    print("Deleting files inside folder " + folder_path)
    num_file_deleted = 0 
    num_folder_deleted = 0

    try:
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if os.path.isdir(fpath): #hapus folder
                print('folder = {}'.format(fpath))
                shutil.rmtree(fpath)
                num_folder_deleted += 1
                pass
            else:
		#hapus file
                print('file = {}'.format(fpath))
                os.remove(fpath)
                num_file_deleted += 1
                pass
        
    except :
        pass
    
    print('file deleted = {}'.format(num_file_deleted))
    print('folder deleted = {}'.format(num_folder_deleted))
def delete_temp_data(main_dir_path):
    for content in os.listdir(main_dir_path):
        content_path = os.path.join(main_dir_path, content)
        if os.path.isdir(content_path):
            for filename in os.listdir(content_path):
                file_path = os.path.join(content_path, filename)
                print(file_path)
                try:
                    if os.path.isdir(file_path) and content.split('_')[0]=='temp':
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete {}. Reason: {}'.format(file_path, e))

def search_file(filepath, data_format):
    temp = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.split('.')[-1] == data_format:
                temp.append(file)
    return temp

def search_unique(target_path):
    target_file = search_file(target_path, 'tif')
    target_list = []
    for target in target_file:
        target_str = '_'.join(target.split('.')[0].split('_')[0:4])
        target_list.append(target_str)
    target_unique = list(set(target_list))
    
    return target_unique

def create_raster_index(composite_raster, out_path, target_unique):
    raster_index = ['MBI', 'MNDWI', 'NDVI', 'SAVI']
    print('[START CREATING RASTER INDEX : {}]'.format(dt.now()).center(50, ' '))
    processing_start_time = time.time()
    for in_raster in composite_raster:
        if in_raster.split('.')[0] not in target_unique:
            print('[INFO] Start Creating Index Raster for {}...'.format(in_raster))
            for index in raster_index:
                if index == 'NDVI':
                    try:
                        NDVI_index = arcpy.sa.RasterCalculator([in_raster+'\Band_7', in_raster+'\Band_3'], ['x','y'], '(x-y)/(x+y)')
                        NDVI_index.save(os.path.join(out_path, '{}_{}.tif'.format(in_raster.split('.')[0],index)))
                        print('[SUCCESS] NDVI has been created')
                    except Exception as e:
                        print('[FAILED] NDVI failed to create. {}'.format(e))
                elif index == 'MNDWI':
                    try:
                        MNDWI_index = arcpy.sa.RasterCalculator([in_raster+'\Band_2', in_raster+'\Band_8'], ['x','y'], '(x-y)/(x+y)')
                        MNDWI_index.save(os.path.join(out_path, '{}_{}.tif'.format(in_raster.split('.')[0],index)))
                        print('[SUCCESS] MNDWI has been created')
                    except Exception as e:
                        print('[FAILED] MNDWI failed to create. {}'.format(e))
                elif index == 'SAVI':
                    try:
                        SAVI_index = arcpy.sa.RasterCalculator([in_raster+'\Band_7', in_raster+'\Band_3'], ['x','y'], '((x-y)/(x+y+0.5))*(1+0.5)')
                        SAVI_index.save(os.path.join(out_path, '{}_{}.tif'.format(in_raster.split('.')[0],index)))
                        print('[SUCCESS] SAVI has been created')
                    except Exception as e:
                        print('[FAILED] SAVI failed to create. {}'.format(e))
                elif index == 'MBI':
                    try:
                        MBI_index = arcpy.sa.RasterCalculator([in_raster+'\Band_8', in_raster+'\Band_9', in_raster+'\Band_7'], ['x','y','z'], '((x-y-z)/(x+y+z))+0.5')
                        MBI_index.save(os.path.join(out_path, '{}_{}.tif'.format(in_raster.split('.')[0],index)))
                        print('[SUCCESS] MBI has been created')
                    except Exception as e:
                        print('[FAILED] MBI failed to create. {}'.format(e))  
    
    elapsed_time = time.time() - init_processing_start_time
    totaltime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Creating Raster Index Total Processing Time: {} \n".format(totaltime))

def inference_raster(composite_raster, raster_composite_index, model_path, out_path, target_unique):
    print('[START INFERENCING IMAGERY DATA : {}]'.format(dt.now()).center(50, ' '))
    processing_start_time = time.time()
    for in_raster in composite_raster:
        if in_raster.split('.')[0] not in target_unique:
            try:
                for raster in raster_composite_index:
                    # Get raster information
                    if raster.split('.')[0] == in_raster.split('.')[0]+'_MBI':
                        MBI_Ras = arcpy.Raster(raster)
                    elif raster.split('.')[0] == in_raster.split('.')[0]+'_MNDWI':
                        MNDWI_Ras = arcpy.Raster(raster)
                    elif raster.split('.')[0] == in_raster.split('.')[0]+'_NDVI':
                        NDVI_Ras = arcpy.Raster(raster)
                    elif raster.split('.')[0] == in_raster.split('.')[0]+'_SAVI':
                        SAVI_Ras = arcpy.Raster(raster)

                lowerLeft = arcpy.Point(MBI_Ras.extent.XMin,MBI_Ras.extent.YMin)
                cellSize = MBI_Ras.meanCellWidth

                # Convert raster to numpy array
                MBI_arr = arcpy.RasterToNumPyArray(MBI_Ras, nodata_to_value=-99)
                MNDWI_arr = arcpy.RasterToNumPyArray(MNDWI_Ras, nodata_to_value=-99)
                NDVI_arr = arcpy.RasterToNumPyArray(NDVI_Ras, nodata_to_value=-99)
                SAVI_arr = arcpy.RasterToNumPyArray(SAVI_Ras, nodata_to_value=-99)

                # Flatten array
                MBI_rav = MBI_arr.ravel()
                MNDWI_rav = MNDWI_arr.ravel()
                NDVI_rav = NDVI_arr.ravel()
                SAVI_rav = SAVI_arr.ravel()

                # Stack array
                stack_arr = np.column_stack((MBI_rav, MNDWI_rav, NDVI_rav, SAVI_rav))

                # Prepare test data
                X_test = pd.DataFrame(stack_arr, columns=['MBI', 'MNDWI', 'NDVI', 'SAVI'])

                # Load model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                # Predict
                y_test = model.predict(X_test)

                # Reshape predicted result
                y_test_reshape = y_test.reshape(MBI_arr.shape)

                # Numpy array to Raster
                newRaster = arcpy.NumPyArrayToRaster(y_test_reshape, lowerLeft, cellSize, value_to_nodata=0)

                # Save raster
                print('[INFO] Start Creating Predicted Raster for {}...'.format(in_raster))
                newRaster.save(os.path.join(out_path, '{}_PREDICTED_RASTER.tif'.format(in_raster.split('.')[0])))
                print('[SUCCESS] Predicted raster has been created')
            except Exception as e:
                print('[FAILED] Predicted raster failed to create. {}'.format(e))
                
    elapsed_time = time.time() - init_processing_start_time
    totaltime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Inferencing Imagery Data Total Processing Time: {} \n".format(totaltime))

def define_projection(composite_raster, predicted_raster):
    print('[START DEFINING PROJECTION : {}]'.format(dt.now()).center(50, ' '))
    processing_start_time = time.time()
    for in_raster in composite_raster:
        arcpy.env.workspace = os.path.join(crawling_path, 'temp_composite_raster')
        dsc = arcpy.Describe(in_raster)
        coord_sys = dsc.spatialReference
        for pred_ras in predicted_raster:
            if pred_ras.split('.')[0] == in_raster.split('.')[0]+'_PREDICTED_RASTER':
                try:
                    print('[INFO] Start Defining Projection for {}...'.format(pred_ras))
                    arcpy.env.workspace = os.path.join(infer_path, 'temp_predicted_raster')
                    arcpy.DefineProjection_management(pred_ras, coord_sys)
                    print('[SUCCESS] {} projected to {}'.format(pred_ras, coord_sys.name))
                except Exception as e:
                    print('[FAILED] {}'.format(e))
                    
    elapsed_time = time.time() - init_processing_start_time
    totaltime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Defining Projection Total Processing Time: {} \n".format(totaltime))

def raster_to_polygon(predicted_raster, out_path, extracted_polygon_temp):
    print('[START CONVERTING RASTER TO POLYGON : {}]'.format(dt.now()).center(50, ' '))
    processing_start_time = time.time()
    for pred_ras in predicted_raster:
        if pred_ras.replace('RASTER.tif', 'POLYGON.shp') not in os.listdir(out_path):
            try:
                print('[INFO] Start Converting Raster to Polygon for {}...'.format(pred_ras))
                out_name = os.path.join(out_path, pred_ras.replace('RASTER', 'POLYGON'))
                arcpy.RasterToPolygon_conversion(pred_ras, out_name, "NO_SIMPLIFY", "VALUE")
                print('[SUCCESS] {} converted to {}'.format(pred_ras, pred_ras.replace('RASTER.tif', 'POLYGON.shp')))

                # Cleansing
                print('[INFO] Start Cleansing for {}...'.format(pred_ras.replace('RASTER.tif', 'POLYGON.shp')))
                success_cleansing = False
                while not success_cleansing:
                    try:
                        sdf = GeoAccessor.from_featureclass(os.path.join(infer_path, 'temp_predicted_polygon', pred_ras.replace('RASTER.tif', 'POLYGON.shp')))
                        sdf = sdf.loc[sdf['gridcode']==1]
                        tanggal = pred_ras.split('_')[2]
                        sdf['img_date'] = dt.strptime(tanggal, '%Y%m%d')
                        sdf['proc_date'] = dt.today()
                        sdf = sdf.reset_index()
                        output_data = os.path.join(infer_path, 'temp_filtered_polygon', pred_ras.replace('RASTER.tif', 'POLYGON.shp'))
                        GeoAccessor.to_featureclass(sdf.spatial, location=output_data, overwrite=True)
                        success_cleansing = True
                    except:
                        pass
                print('[SUCCESS], continue to append to temp')
                arcpy.management.Append(output_data, extracted_polygon_temp, "NO_TEST", r'id "id" true true false 80 Text 0 0,First,#,output_data,Id,-1,-1;processeddate "date" true true false 8 Date 0 0,First,#,output_data,proc_date,-1,-1;imagerydate "imagerydate" true true false 8 Date 0 0,First,#,output_data,img_date,-1,-1;total_devegetasi "total_devegetasi" true true false 8 Double 0 0,First,#', '', '')
                print('[SUCCESS], Done')
            except Exception as e:
                print('[FAILED] {}'.format(e))  
                
    elapsed_time = time.time() - init_processing_start_time
    totaltime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Converting Raster to Polygon Total Processing Time: {} \n".format(totaltime))

#Clean Existing Data:
arcpy.management.TruncateTable(extracted_polygon_temp)
delete_files(os.path.join(infer_path, 'temp_raster_index'))
delete_files(os.path.join(infer_path, 'temp_predicted_raster'))
delete_files(os.path.join(infer_path, 'temp_predicted_polygon'))
delete_files(os.path.join(infer_path, 'temp_filtered_polygon'))

# Set Parameter Indexing
filepath = os.path.join(crawling_path, 'temp_composite_raster')
composite_raster = search_file(filepath, 'tif')
arcpy.env.workspace = os.path.join(crawling_path, 'temp_composite_raster')
out_path = os.path.join(infer_path, 'temp_raster_index')

# Creating Raster Index
target_unique = search_unique(out_path)
create_raster_index(composite_raster, out_path, target_unique)

# Set Parameter Inferencing
filepath = os.path.join(infer_path, 'temp_raster_index')
raster_composite_index = search_file(filepath, 'tif')
arcpy.env.workspace = os.path.join(infer_path, 'temp_raster_index')
model_path = os.path.join(infer_path, 'model', 'model_logreg.sav')
out_path = os.path.join(infer_path, 'temp_predicted_raster')

# Inferencing Raster
target_unique = search_unique(out_path)
inference_raster(composite_raster, raster_composite_index, model_path, out_path, target_unique)

# Set Parameter Define Projection
filepath = os.path.join(infer_path, 'temp_predicted_raster')
predicted_raster = search_file(filepath, 'tif')

# Defining Projection
define_projection(composite_raster, predicted_raster)

# Set Parameter Conversion
arcpy.env.workspace = os.path.join(infer_path, 'temp_predicted_raster')
out_path = os.path.join(infer_path, 'temp_predicted_polygon')

# Converting Raster to Polygon
raster_to_polygon(predicted_raster, out_path, extracted_polygon_temp)

#Calculate geometry and filter
arcpy.management.CalculateGeometryAttributes(extracted_polygon_temp, "total_devegetasi AREA_GEODESIC", '', "HECTARES", 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]', "SAME_AS_INPUT")
arcpy.analysis.Select(extracted_polygon_temp, extracted_polygon_temp_filtered, "total_devegetasi < 20000")


elapsed_time = time.time() - init_processing_start_time
totaltime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print("Inferencing Imagery Total Processing Time: {} \n".format(totaltime))
print("Finished inferencing process")
