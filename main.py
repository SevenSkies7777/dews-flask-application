from pathlib import Path

from flask import Flask, jsonify

from Rainfall_data_extraction import RainfallDataProcessor

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Milk Production API!"

@app.route("/service-api/v1/rainfall-data/process-rainfall-data", methods=["GET"])
def process_rainfall_data():
    try:
        # Initialize the processor
        processor = RainfallDataProcessor(
            db_user="root",
            db_password="*Database630803240081",
            db_host="127.0.0.1",
            db_name="livelihoodzones"
        )

        # Define file paths relative to the project root
        nc_file_path = Path("Pr.nc").resolve()
        shapefile_path = Path("GADM/gadm41_KEN_3.shp").resolve()

        # Process rainfall data
        processor.processRainfallData(nc_file_path, shapefile_path, specific_wards=None)

        # Close the connection
        processor.close()

        # Return a valid JSON response
        return jsonify({"message": "Rainfall data processing completed successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=6060)