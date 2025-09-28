Drone SRT & Chainage Tool
This Streamlit application provides a comprehensive workflow for processing drone flight data from SRT files and calculating their chainage relative to a KML polyline. This upgraded version introduces robust features for handling multiple flights, ensuring accurate projections, and providing detailed outputs for professional surveying and GIS analysis.

Creator: Vijay Parmar — for BGol (Community of Surveyors and GIS Experts)

Key Features
Multi-SRT File Handling: Upload and process multiple SRT files at once.

Automatic Merging: SRT files are automatically sorted by their first timestamp and merged into a single, continuous flight track for chainage computation.

Manual Reordering: A user-friendly interface allows you to manually reorder the SRT files before merging.

Always-Project Chainage: Every drone sample is projected to the closest point on the chainage polyline, eliminating the need for distance thresholds and providing a chainage value for every point.

Perpendicular Distance Calculation: The tool calculates and exports the perpendicular distance (in meters) from each drone point to its projected point on the chain, allowing for easy quality assessment.

High-Accuracy Projections: Utilizes pyproj and shapely with an Azimuthal Equidistant projection for meter-accurate planar calculations. A fallback method using Haversine is available if these libraries are not installed.

Colored KML Exports: Generates a single KML file containing the chainage line and all individual drone flights, each colored differently for easy visual distinction. Start and end chainage markers are included for each flight.

Comprehensive Outputs: Download a ZIP file containing:

merged_track.csv and merged_track.srt: The combined data for all flights.

drone_flights.kml: The visual KML output.

Per-flight CSV and SRT files in a separate folder.

Interactive 3D Preview: Visualize the chainage polyline and all drone flight paths in an interactive 3D plot directly in the app.

Installation & Usage
Clone the repository or download the files.

Install dependencies: It is highly recommended to use a virtual environment. For the best performance and accuracy, install all required libraries from requirements.txt.

pip install -r requirements.txt

Note: If pyproj and shapely are not installed, the application will run in a fallback mode with lower accuracy and display a warning.

Run the Streamlit application:

streamlit run streamlit_app.py

Follow the steps in the UI:

Step 1: Upload your chainage KML file and one or more drone SRT files.

Step 2: Configure the KML. You can swap the start/end direction of the chainage polyline.

Step 3: Order the SRT flights. They are auto-sorted by timestamp, but you can manually reorder them using the up/down arrows.

Step 4: Click "Compute Chainage" to process the data.

Step 5 & 6: Preview the results in the table and 3D map, then download your output files.

Understanding the Outputs
chainage_km: The distance along the KML polyline to the projected point, measured in kilometers with 3 decimal places (e.g., 1.234 km).

distance_to_chain_m: The perpendicular (shortest) distance from the drone's position to the KML polyline, measured in meters with 2 decimal places. This value is crucial for assessing how far off the flight path was from the chainage line.

The per-flight exports allow you to analyze the distance_to_chain_m for each individual flight, helping to identify any flights that deviated significantly from the intended path.