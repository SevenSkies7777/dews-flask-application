from pathlib import Path

from flask import Flask, jsonify

from HHA_Outliers_Detection_Service import process_outliers
from Rainfall_data_extraction import RainfallDataProcessor
from flask import request
from Milk_Production_Forecasting_Main import process_milk_production_forecasts
from Prediction_Residual_Plots import process_residual_plots
from Grazing_Dist_Forecasting_Main import process_grazing_distance_forecasts

app = Flask(__name__)


@app.route("/")
def home():
  return "Welcome to the Milk Production API!"


@app.route("/service-api/v1/rainfall-data/process-rainfall-data",
           methods=["GET"])
def process_rainfall_data():
  try:
    # Initialize the processor
    processor = RainfallDataProcessor(
        db_user="root",
        db_password="*Database630803240081",
        db_host="127.0.0.1",
        db_name="dews_machine_learning"
    )

    # Define file paths relative to the project root
    nc_file_path = Path("Pr.nc").resolve()
    shapefile_path = Path("new_livelihood_zones/new_livelihood_zones.shp").resolve()

    # Process rainfall data
    processor.processRainfallData(nc_file_path, shapefile_path,
                                  specific_wards=None)

    # Close the connection
    processor.close()

    # Return a valid JSON response
    return jsonify(
        {"message": "Rainfall data processing completed successfully"}), 200

  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/service-api/v1/outliers/process/outliers-by-county",
             methods=["GET"])
def process_outliers_by_county():
    try:
      # Get query parameters from the request
      county_id = request.args.get("countyId", type=int)
      data_collection_exercise_id = request.args.get("dataCollectionExerciseId",
                                                     type=int)


      # Call the outlier processing function
      process_outliers(county_id, data_collection_exercise_id)

      # Return a valid JSON response
      return jsonify(
          {"message": "Outlier processing completed successfully"}), 200

    except Exception as e:
      return jsonify({"error": str(e)}), 500

@app.route("/service-api/v1/predictions/process/milk-predictions",
             methods=["GET"])
def process_milk_predictions():
    try:
      county_id = request.args.get("countyId", type=int)

      # Call the outlier processing function
      process_milk_production_forecasts(county_id)

      process_grazing_distance_forecasts(county_id)

      process_residual_plots(county_id, "TotalDailyQntyMilkedInLtrs")

      process_residual_plots(county_id, "DistInKmsToWaterSourceFromGrazingArea")

      # Return a valid JSON response
      return jsonify(
          {"message": "Milk predictions processing completed successfully"}), 200

    except Exception as e:
      return jsonify({"error": str(e)}), 500

@app.route("/service-api/v1/predictions/process/residual-plots",
             methods=["GET"])
def process_residual_plots_api():
    try:
      county_id = request.args.get("countyId", type=int)
      indicator = request.args.get("indicator", type=str)

      process_residual_plots(county_id, indicator)

      # Return a valid JSON response
      return jsonify(
          {"message": "Residual plots processing completed successfully"}), 200

    except Exception as e:
      return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(debug=False, threaded=True, host="0.0.0.0", port=6060)
