import cv2
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform, linemerge
import shapely
import math
import argparse

def polar_to_cartesian(radius, angle):
    """
    Converts polar coordinates (radius, angle) to Cartesian coordinates (x, y).
    """
    x = radius * math.cos(math.radians(angle))
    y = radius * math.sin(math.radians(angle))
    return (x, y)

def read_coords_from_file(file_path):
    """
    Reads polar coordinates from a file.

    Args:
        file_path: Path to the input file containing polar coordinates.

    Returns:
        A list of tuples, where each tuple represents a x,y point.
    """

    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            xvalue, yvalue = map(float, line.strip().split(','))
            coords.append((xvalue, yvalue))

    return coords

def convert_to_polar(points):
    # 1. Find center
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    minX, maxX = min(x_coords), max(x_coords)
    minY, maxY = min(y_coords), max(y_coords)
    centerX = (minX + maxX) / 2
    centerY = (minY + maxY) / 2
    print(centerX, centerY)
    polar_points = []
    max_radius = 0

    for x, y in points:
        # 2. Relative Cartesian
        relX = x - centerX
        relY = y - centerY

        # 3. Polar
        radius = round(math.sqrt(relX**2 + relY**2))
        angle_rad = math.atan2(relY, relX)

        # 4. Angle to degrees
        angle_deg = round(math.degrees(angle_rad), 2)

        max_radius = max(max_radius, radius)

    print("max radius: ", max_radius)

    for x, y in points:
        # 2. Relative Cartesian
        relX = x - centerX
        relY = y - centerY

        # 3. Polar
        radius = math.sqrt(relX**2 + relY**2)
        scaled_radius = round((radius / max_radius) * 80)
        angle_rad = math.atan2(relY, relX)

        # 4. Angle to degrees
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        #polar_points.append((scaled_radius, round(angle_deg, 2)))
        polar_points.append((scaled_radius, round(angle_deg)))

    return polar_points

def improved_remove_close_non_consecutive_points(polar_coords, close_threshold_factor=0.1, indice_threshold=5):
    """
    Removes non-consecutive points that are too close to each other.

    Args:
        polar_coords: A list of tuples, where each tuple represents a polar point
            (radius, angle in degrees).
        close_threshold_factor: A factor to multiply the average distance by to determine the close_threshold.
        indice_threshold: The minimum number of indices between points to be considered for removal.

    Returns:
        A new list of polar coordinates with close non-consecutive points removed.
    """
    # Calculate average distance between consecutive points
    distances = [math.dist(polar_coords[i], polar_coords[i + 1]) for i in range(len(polar_coords) - 1)]
    avg_distance = sum(distances) / len(distances) if distances else 0

    # Calculate adaptive close_threshold
    close_threshold = avg_distance * close_threshold_factor

    filtered_coords = []
    for i in range(len(polar_coords)):
        keep_point = True
        for j in range(i + indice_threshold, len(polar_coords)):
            # Calculate angular distance
            angular_distance = min(abs(polar_coords[i][1] - polar_coords[j][1]), 360 - abs(polar_coords[i][1] - polar_coords[j][1]))

            # Check if points are too close
            if angular_distance < close_threshold:
                keep_point = False
                break

        if keep_point:
            filtered_coords.append(polar_coords[i])

    return filtered_coords

def remove_close_non_consecutive_points(polar_coords, close_threshold, indice_threshold):
    """
    Removes non-consecutive points that are too close to each other.

    Args:
        polar_coords: A list of tuples, where each tuple represents a polar point
            (radius, angle in degrees).
        threshold: The distance threshold for considering points as "too close".

    Returns:
        A new list of polar coordinates with close non-consecutive points removed.
    """

    filtered_coords = []
    for i in range(len(polar_coords)):
        keep_point = True
        for j in range(i + indice_threshold, len(polar_coords)):  # Check points more than 10 indices away
            # Calculate distance between points i and j
            distance = math.sqrt(
                (polar_coords[i][0] - polar_coords[j][0])**2 +
                (polar_coords[i][1] - polar_coords[j][1])**2
            )
            if distance < close_threshold:
                keep_point = False  # Mark point i for removal if too close
                break  # No need to check further for this point

        if keep_point:
            filtered_coords.append(polar_coords[i])

    return filtered_coords



def create_continuous_trace_path(external_contour, inner_contours):
    """
    Creates a continuous trace path for a plotter, splicing in inner contours.
    Starts at the closest external point to the start/end of each inner contour.

    Args:
        external_contour: A NumPy array representing the external contour.
        inner_contours: A list of NumPy arrays, each representing an inner contour.

    Returns:
        A list of (x, y) coordinates representing the continuous trace path.
    """

    if len(external_contour) < 3:
        raise ValueError("External contour must have at least 3 points to create a Polygon.")

    external_polygon = Polygon(external_contour.reshape(-1, 2))

    # Create LineStrings for all inner contours that are greater than 3 (ignores small ones)
    inner_linestrings = [LineString(contour.reshape(-1, 2)) for contour in inner_contours if contour.reshape(-1, 2).shape[0] > 3]
    for i in range(len(inner_linestrings)):
        # Strip last coordinate as most Linestrings are closing weirdly.  Come back and fix another way
        inner_linestrings[i] = LineString(inner_linestrings[i].coords[:-1])



    # Store information about closest points and start/end for each inner contour
    inner_contour_info = []
    for linestring in inner_linestrings:
        #print("Inner LineString Coords: ", linestring)
        start_point = linestring.coords[0]
        end_point = linestring.coords[-1]

        # Find closest point on external contour for start and end points
        closest_point_start = external_polygon.exterior.interpolate(
            external_polygon.exterior.project(Point(start_point))
        )
        closest_point_end = external_polygon.exterior.interpolate(
            external_polygon.exterior.project(Point(end_point))
        )

        # Determine which is closer (start or end)
        distance_start = Point(start_point).distance(closest_point_start)
        distance_end = Point(end_point).distance(closest_point_end)

        if distance_start < distance_end:
            closest_point = closest_point_start
            is_start = True
        else:
            closest_point = closest_point_end
            is_start = False

        # Find index of closest external point
        closest_external_index = np.argmin(
            [np.linalg.norm(np.array(closest_point.coords[0]) - np.array(point[0]))
             for point in external_contour]
        )
        closest_external_index = min(closest_external_index, len(external_contour) - 1)  # Clamp index

        inner_contour_info.append({
            'linestring': linestring,
            'closest_external_index': closest_external_index,
            'is_start': is_start
        })

    # Sort inner contours based on closest_external_index
    #inner_contour_info.sort(key=lambda x: x['closest_external_index'])
    seen_indices = set()  # Track indices already encountered
    for i, info in enumerate(inner_contour_info):
      while info['closest_external_index'] in seen_indices:
        info['closest_external_index'] += 1  # Increment until unique
      seen_indices.add(info['closest_external_index'])  # Mark index as seen


    # Build the continuous trace path
    trace_path = []
    completed_inner_contours = set()  # Keep track of completed inner contours

    trace_back_distance_threshold = 3

    for i, point in enumerate(external_contour):
        trace_path.append(tuple(point[0]))  # Add current external point

        # Check for inner contour insertion
        for info in inner_contour_info:
            if info['closest_external_index'] == i and info['linestring'] not in completed_inner_contours:
                # Splice in the inner contour (LineString)
                if info['is_start']:
                    trace_path.extend(list(info['linestring'].coords))
                    distance = np.linalg.norm(np.array(info['linestring'].coords[-1]) - np.array(point[0]))  # point[0] is external point
                    if distance > trace_back_distance_threshold:
                      trace_path.extend(list(info['linestring'].coords)[-2::-1]) # reverse trace back along inner contour to external contour point
                    else:
                      trace_path.extend(list(info['linestring'].coords[0]))
                else:
                    trace_path.extend(list(info['linestring'].coords)[::-1])  # Reverse order for LineString
                    distance = np.linalg.norm(np.array(info['linestring'].coords[0]) - np.array(point[0]))  # point[0] is external point
                    if distance > trace_back_distance_threshold:
                      trace_path.extend(list(info['linestring'].coords)[1::-1]) # reverse trace back along inner contour to external contour point
                    else:
                      trace_path.extend(list(info['linestring'].coords)[-1])

                completed_inner_contours.add(info['linestring'])  # Mark as completed
                trace_path.append(tuple(point[0]))  # Return to the external contour point
                break  # Move to the next external point

    return trace_path




def split_contour_by_distance(contour, distance_threshold):
    """
    Splits a contour into multiple contours based on the distance between consecutive points.
    """

    split_contours = []
    new_contour = [contour[0]]

    for i in range(1, len(contour)):
        distance = np.linalg.norm(contour[i][0] - contour[i - 1][0])

        if distance > distance_threshold:
            split_contours.append(np.array(new_contour))
            new_contour = []
            new_contour.append(contour[i])
        else:
            new_contour.append(contour[i])


    split_contours.append(np.array(new_contour))

    return split_contours


def trace_image(image_path, simple_mode=False):
  """
  Traces the outline of a black and white image and returns a list of coordinates.

  Args:
    image_path: The path to the image file.

  Returns:
    A list of (x, y) coordinates representing the traced path.
    Returns None if the image cannot be processed.
  """
  try:
    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    top, bottom, left, right = 10, 10, 10, 10

    #Add white border to image
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    img = cv2.flip(img, 0)


    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Threshold the image to create a binary image (black and white)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # Invert for easier contour detection
    #thinned_image = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Find contours in the image
    external_contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not external_contour:
        print("No contours found in the image.")
        return None
    
    selected_external_contour = max(external_contour, key=cv2.contourArea)
    #contours, hierarchy = cv2.findContours(thinned_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found in the image.")
        return None

    print("Number of contours found:", len(contours))
    print("number of external contours found: ", len(external_contour))


    external_contour_area = cv2.contourArea(selected_external_contour)  # Assuming only one external contour
    filtered_contours = []
    for i, contour in enumerate(contours):
      # Check if the contour has a parent (enclosed) and is less than 30% of the external contour area
      if hierarchy[0][i][3] != -1 and cv2.contourArea(contour) < 0.3 * external_contour_area:
        filtered_contours.append(contour)

    print("Number of filtered contours: ", len(filtered_contours))
    for i, contour in enumerate(filtered_contours):
      area = cv2.contourArea(contour)
      print(f"Contour {i}: Area = {area}")
    print("Area of external Contour: ", external_contour_area)

    threshold_distance = 10
    splitting_distance_threshold = 5

    stripped_contours = []
    for contour in filtered_contours:
      stripped_contour = []

      for point in contour:
          point_tuple = tuple(int(coord) for coord in point[0])
          dist_to_external = cv2.pointPolygonTest(external_contour[0], point_tuple, True)

          # If the distance is greater than the threshold (point is outside the external contour)
          if dist_to_external > threshold_distance:
              stripped_contour.append(point)
    
      split_contours = []  # Initialize to an empty list
      
      # Code to split the stripped_contour
      if stripped_contour:
          split_contours = split_contour_by_distance(np.array(stripped_contour), splitting_distance_threshold)

      if split_contours:
        if len(split_contours) > 1:
          print("Number of split contours: ", len(split_contours))
          for i, split_contour in enumerate(split_contours):
            length = len(split_contour)
            print(f"Split Contour {i}: Length = {length}")

        for split_contour in split_contours:
          if len(stripped_contour) > 9:
            #stripped_contours.append(np.array(stripped_contour).reshape(-1, 1, 2))  # Reshape here
            stripped_contours.append(np.array(split_contour.reshape(-1, 1, 2)))  # Reshape here

      else:
          print("Warning: Stripped contour is empty, skipping.")
    

    ### SIMPLE MODE LOGIC 
    distance_threshold = 15  # Define your distance threshold here
    final_stripped_contours = []

    if simple_mode:
      for stripped_contour in stripped_contours:
        # Check if ANY point in the contour is within the distance threshold
        distances_from_external = []
        for point in stripped_contour:
          point_tuple = tuple(int(coord) for coord in point[0])
          dist_to_external = cv2.pointPolygonTest(external_contour[0], point_tuple, True)
          distances_from_external.append(round(dist_to_external))
          
        print("Distance array: ", distances_from_external)
        min_distance = min(distances_from_external)
        if min_distance < distance_threshold:
          final_stripped_contours.append(stripped_contour)
          continue
        

    else:
      final_stripped_contours = stripped_contours


    

    print("External Contour Length: ", len(selected_external_contour))
    print("Number of final inner contours: ", len(final_stripped_contours))
    for i, contour in enumerate(final_stripped_contours):
      length = len(contour)
      print(f"Inner Contour {i}: Length = {length}")

    #simplify external_contour   
    if len(selected_external_contour) >= 3:  # Check if enough points for a polygon   
        external_polygon = Polygon(selected_external_contour.reshape(-1, 2))
        if len(external_polygon.exterior.coords) > 6:
            simplified_external_polygon = external_polygon.simplify(tolerance=1)  # Adjust tolerance as needed
        else:
            simplified_external_polygon = external_polygon  # Keep the original polygon
    else:
        return None

     #If simplified to a linestring, get coords, otherwise get exterior coords
    if isinstance(simplified_external_polygon, LineString):
      simplified_coords = np.array(simplified_external_polygon.coords)
    else:
      simplified_coords = np.array(simplified_external_polygon.exterior.coords)

    external_contour_simplified = simplified_coords.reshape(-1, 1, 2)
  
                     
    inner_contours_simplified = []
    for contour in final_stripped_contours:
        # Convert the contour to a Shapely LineString or Polygon
        if len(contour) >= 3:  # Check if enough points for a polygon
            if contour[0][0][0] == contour[-1][0][0] and contour[0][0][1] == contour[-1][0][1]:  # Check if closed
                polygon = Polygon(contour.reshape(-1, 2))  # If closed and has 3+ points, make polygon
                # Simplify the polygon

                #simplified_polygon = polygon.simplify(tolerance=1)  # Adjust tolerance as needed
                if len(polygon.exterior.coords) > 6:
                    simplified_polygon = polygon.simplify(tolerance=1)  # Adjust tolerance as needed
                else:
                    simplified_polygon = polygon  # Keep the original polygon

                # If simplified to a linestring, get coords, otherwise get exterior coords
                if isinstance(simplified_polygon, LineString):
                    simplified_coords = np.array(simplified_polygon.coords)
                else:
                    simplified_coords = np.array(simplified_polygon.exterior.coords)

                inner_contours_simplified.append(simplified_coords.reshape(-1, 1, 2))
            else:
                # If not closed but has 3+ points, make LineString and simplify
                linestring = LineString(contour.reshape(-1, 2))
                if linestring.length > 6:
                    simplified_linestring = linestring.simplify(tolerance=1)  # Adjust tolerance as needed
                else:
                    simplified_linestring = linestring  # Keep the original polygon

                #simplified_linestring = linestring.simplify(tolerance=1)
                #print("Simplified Line String: ", simplified_linestring)
                inner_contours_simplified.append(np.array(simplified_linestring.coords).reshape(-1, 1, 2))
        elif len(contour) > 1:
            # If 2 points, make LineString (can't simplify)
            linestring = LineString(contour.reshape(-1, 2))
            inner_contours_simplified.append(np.array(linestring.coords).reshape(-1, 1, 2))
        else:
            # If single point or empty, add as is
            inner_contours_simplified.append(contour)

    print("Length of simplified external contour: ", len(external_contour_simplified))
    print("Number of final inner contours simplified: ", len(inner_contours_simplified))
    total_number_of_points = len(external_contour_simplified)
    for i, contour in enumerate(inner_contours_simplified):
      length = len(contour)
      total_number_of_points += length
      print(f"Contour {i}: Length = {length}")

    print("Total number of points: ", total_number_of_points)

    

    return selected_external_contour, external_contour_simplified, inner_contours_simplified
  except Exception as e:
    print(f"An error occurred: {e}")
    return None




##################################################
#Executing Code Start

def main():
    parser = argparse.ArgumentParser(description='Process an image and convert to polar coordinates.')
    parser.add_argument('--simple_mode', action='store_true', help='Use simple mode for tracing the image')
    parser.add_argument('--do_plots', action='store_true', help='Generate and save plots')
    parser.add_argument('--external_mode', action='store_true', help='Use external mode for tracing only the external outline of the image')
    # Parse arguments
    args = parser.parse_args()
    print("Simple Mode: ", args.simple_mode)
    print("Do Plots: ", args.do_plots)
    print("External Mode: ", args.external_mode)


    image_file = 'sand_garden_image.png'  # Make sure this file is present or change accordingly
    
   
    
    externalcontour, external_contour_simplified, inner_contours_simplified = trace_image(image_file, simple_mode=args.simple_mode)

    # Assuming you have 'external_contour' and 'inner_contours' from your previous processing
    if args.external_mode:
        print("External Mode: Only returns points for outer contour")
        continuous_path = [tuple(point[0]) for point in external_contour_simplified]
    else:
        continuous_path = create_continuous_trace_path(external_contour_simplified, inner_contours_simplified)

    # Save the coordinates to a file
    #with open('coordinates.txt', 'w') as f:
    #for x, y in continuous_path:
        #f.write(f"{x},{y}\n")
    #with open('coordinates.txt', 'w') as f:
    # for x, y in coordinates:
    #  f.write(f"{x},{y}\n")

    print("Length of continous path: ", len(continuous_path))
    print("Type of continous path: ", type(continuous_path))
   
    polar_coords = convert_to_polar(continuous_path)

    template_file_path = 'drawing_array_template.h'
    with open(template_file_path, 'r') as template_file:
        template_content = template_file.read()

    # Format the polar coordinates
    formatted_points = ',\n'.join([f"{{{point[0]}, {point[1]}}}" for point in polar_coords])

    # Replace the <points> tag with the formatted points
    new_content = template_content.replace('<points>', formatted_points)

    # Save the new content to drawing_array.h
    output_file_path = 'src/drawing_array.h'
    with open(output_file_path, 'w') as output_file:
        output_file.write(new_content)

    print(f"New drawing array saved to {output_file_path}")


    with open("polarCoords.txt", "w") as file:
        for point in polar_coords:
            file.write(f"{{{point[0]}, {point[1]}}},\n")  # Add brackets here





    ###################################################################################################
    #PLOTS BELOW
    if args.do_plots:
        fig0, ax0 = plt.subplots()
        num_contours = len(externalcontour)  # Changed to externalcontour
        colors = plt.cm.get_cmap('viridis', num_contours)  # Get a colormap

        for i, contour in enumerate(externalcontour):
            # Convert contour to NumPy array for plotting
            contour_array = np.array(externalcontour)

            # Reshape contour_array to (number_of_points, 2)
            contour_array = contour_array.reshape(-1, 2)

            # Extract x and y coordinates
            x_coords = contour_array[:, 0]
            y_coords = contour_array[:, 1]

            ax0.plot(x_coords, y_coords, color=colors(i), label=f'Contour {i+1}')

        # Customize the plot
        ax0.set_title("External Contour")
        ax0.set_xlabel("X-coordinate")
        ax0.set_ylabel("Y-coordinate")
        ax0.legend()  # Show legend with contour labels
        ax0.grid(True)
        fig0.savefig('plots/external_contour_plot.png')


        if not args.external_mode:
            fig2, ax2 = plt.subplots()  # Create fig2, ax2 for the second plot

            # Plot each contour with a different color
            num_final_inner_contours = len(inner_contours_simplified)
            colors = plt.cm.get_cmap('viridis', num_final_inner_contours)  # Get a colormap with num_contours colors

            for i, contour in enumerate(inner_contours_simplified):
                # Convert contour to NumPy array for plotting
                contour_array = np.array(contour)

                # Reshape contour_array to (number_of_points, 2)
                contour_array = contour_array.reshape(-1, 2)

                # Extract x and y coordinates
                x_coords = contour_array[:, 0]
                y_coords = contour_array[:, 1]

                ax2.plot(x_coords, y_coords, color=colors(i), label=f'Contour {i+1}')

            # Customize the plot
            ax2.set_title("Final Simplified Inner Contours with Different Colors")
            ax2.set_xlabel("X-coordinate")
            ax2.set_ylabel("Y-coordinate")
            ax2.legend()  # Show legend with contour labels
            ax2.grid(True)
            fig2.savefig('plots/inner_contours_plot.png')





            # *** Add scatter plot below ***
            fig8, ax8 = plt.subplots()  # Create a new figure and axes

            external_contour_array = np.array(externalcontour)
            external_contour_array = external_contour_array.reshape(-1, 2)
            x_coords = external_contour_array[:, 0]
            y_coords = external_contour_array[:, 1]
            ax8.scatter(x_coords, y_coords, color=colors(i), label=f'Contour {i+1}')
                    
            for i, contour in enumerate(inner_contours_simplified):
                # Convert contour to NumPy array for plotting
                contour_array = np.array(contour)

                # Reshape contour_array to (number_of_points, 2)
                contour_array = contour_array.reshape(-1, 2)

                # Extract x and y coordinates
                x_int_coords = contour_array[:, 0]
                y_int_coords = contour_array[:, 1]

                # Plot as scatter points
                ax8.scatter(x_int_coords, y_int_coords, color=colors(i), label=f'Contour {i+1}')


            # Customize the scatter plot
            ax8.set_title("Final Contours Simplified (Scatter Plot)")
            ax8.set_xlabel("X-coordinate")
            ax8.set_ylabel("Y-coordinate")
            ax8.legend()
            ax8.grid(True)
            fig8.savefig('plots/final_contours_simplified_scatter_plot.png')


        #continuous_path PLOTs

        fig9, ax9 = plt.subplots()
        points_x_cont = [point[0] for point in continuous_path]
        points_y_cont = [point[1] for point in continuous_path]
        ax9.scatter(points_x_cont, points_y_cont, color='r', label='Points')
        ax9.set_title("Continuous Path Points")
        ax9.set_xlabel("X-coordinate")
        ax9.set_ylabel("Y-coordinate")
        ax9.grid(True)
        fig9.savefig('plots/continuous_path_points_plot.png')

        # Figure 4 - Line Connected Plot
        fig10, ax10 = plt.subplots()  # Create a new figure and axes object for Figure 4
        ax10.plot(points_x_cont, points_y_cont, marker='o', linestyle='-', color='b')  # Plot on ax4
        ax10.set_title("Continous Path Line Connected")
        ax10.set_xlabel("X-coordinate")
        ax10.set_ylabel("Y-coordinate")
        ax10.grid(True)
        fig10.savefig('plots/continuous_path_line_connected_plot.png')


        # Figure 5 - Polar Plot
        fig11, ax11 = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})  # Adjust (10, 6) to your desired size


        # Plot the data
        #ax.plot(theta, r, marker='o')
        r = [coord[0] for coord in polar_coords]
        theta = [np.deg2rad(coord[1]) for coord in polar_coords]
        for i, (t, rad) in enumerate(zip(theta, r)):
            ax11.plot(t, rad, marker='o', linestyle='none')
            ax11.text(t, rad, str(i+1), ha='center', va='bottom')  # Label with index + 1

        # Customize the plot (optional)
        ax11.set_title("Polar Plot")
        ax11.grid(True)
        fig11.savefig('plots/polar_plot.png')

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add each plot image to the PDF
        if not args.external_mode:
            plot_files = ['plots/external_contour_plot.png', 'plots/inner_contours_plot.png', 'plots/final_contours_simplified_scatter_plot.png', 'plots/continuous_path_points_plot.png', 'plots/continuous_path_line_connected_plot.png', 'plots/polar_plot.png']  # Add more plot filenames as needed
        else:
            plot_files = ['plots/external_contour_plot.png', 'plots/continuous_path_points_plot.png', 'plots/continuous_path_line_connected_plot.png', 'plots/polar_plot.png']

        for plot_file in plot_files:
            pdf.add_page()
            pdf.image(plot_file, x=10, y=10, w=pdf.w - 20)  # Adjust x, y, and w as needed

        # Save the PDF
        pdf_output_path = 'plots/plots_document.pdf'
        pdf.output(pdf_output_path)

        print(f"Plots saved as images and combined into {pdf_output_path}")


if __name__ == '__main__':
    main()


