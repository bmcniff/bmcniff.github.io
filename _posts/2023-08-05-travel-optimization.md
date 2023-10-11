---
layout: post
title: "Travel Smarter, Not Harder"
date: 2023-08-05 15:13:18 +0200
image: 15.png
tags: [jekyll, docs]
categories: jekyll
---

*Solving for the optimal route through a travel itinerary.*

### Overview

I recently took a trip to Iceland and spent several days in Reykjavík. This was my first time in the city and I was excited to visit a variety of museums, landmarks, and restaurants. However, I had absolutely no understanding of the layout of the city, and wanted to find a strategy to quickly understand how to organize my list of activities.

Admittedly, I need very little motivation to start a new pet project, and this certainly sufficed. My goal at the start of this project was an optimization model that would allow me to input the location of all of the activities I wanted to experience and the length of the trip in days. Ideally, the model would create a daily itinerary and guide me through these activities in as efficient a path as possible.

### Data Processing

The first step in this process is getting a list of activities into the script. I wanted to be able to quickly iterate by adding/removing activities, which meant creating an automated solution for pulling the addresses and latitude/longitude coordinates. More on that in a bit.

For now we can just throw our activities into a list and import some libraries that (in hindsight) I know we'll use later.

``` python
import numpy as np
import pandas as pd
import googlemaps
import json

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import gmplot
```

``` python
# list of business/location names for activities
locations = [
    'The Reykjavik EDITION',
    'Hallgrímskirkja',
    'Sun Voyager',
    'National Museum of Iceland',
    'The Settlement Exhibition',
    'Harpan Concert Hall',
    'Laugavegur Reykjavík',
    'Aurora Reykjavík',
    'The Einar Jónsson Museum',
    'Reykjavík Maritime Museum',
    'Grótta Island Lighthouse',
    'The Icelandic Punk Museum',
    'Mat Bar',
    'DEIG workshop',
    'Anna Jóna',
    'Bæjarins Beztu Pylsur',
    '12 Tónar',
    'Penninn Eymundsson Skólavörðustíg',
    'Hljómskálagarðurinn',
    'Skólavörðustígur']
```

I'm using the googlemaps library to automate the retrieval of business information. I created a function that pulls in the full JSON response (example shown below) and extracts the addresses and lat/long coordinates.

One interesting note is that I found I could only get accurate results by specifying an anchor point in the request. In this case, that meant including the lat/long of the closest city, and the radius to check within (I ended up using 50km).

``` python
client = googlemaps.Client(key='secret_key')
```

``` python
def get_addresses(location_list, city_lat, city_lng):
    ''' refernces Google Maps Places API and adds response for each location to dictionary '''
    location_info = {}
    for location in location_list:
        # query Google Maps for business name within specified city
        location_info[location] = client.places(location=[city_lat, city_lng], radius=50000, query=location)
        
    return location_info

locations_full_info = get_addresses(locations, 64.128288, -21.827774)

# return first example for demonstration
locations_full_info[next(iter(locations_full_info))]
```

    {'html_attributions': [],
     'results': [{'business_status': 'OPERATIONAL',
       'formatted_address': 'Austurbakki 2, 101 Reykjavík, Iceland',
       'geometry': {'location': {'lat': 64.1499243, 'lng': -21.9353028},
        'viewport': {'northeast': {'lat': 64.15134187989271,
          'lng': -21.93413812010727},
         'southwest': {'lat': 64.14864222010726, 'lng': -21.93683777989272}}},
       'icon': 'https://maps.gstatic.com/mapfiles/place_api/icons/v1/png_71/lodging-71.png',
       'icon_background_color': '#909CE1',
       'icon_mask_base_uri': 'https://maps.gstatic.com/mapfiles/place_api/icons/v2/hotel_pinlet',
       'name': 'The Reykjavik EDITION',
       'photos': [{'height': 811,
         'html_attributions': ['<a href="https://maps.google.com/maps/contrib/109037278140407822583">A Google User</a>'],
         'photo_reference': 'ATJ83zjjf-GzaT_zTYhxgztk9AblInwVdl_64hYhTxgcmBMM_gWWFVpUOnwpwPFw-_zsblXVp3pVeovlUQEphzi2lNp2PHBVLqdPVyT9tRdb5_hwnG4aRnGCPSBlM8qZoX6Boy3aWVm7rdC9PK_rdN7S9tgezMz-msQvXiEuy_xb6kgsYzHh',
         'width': 1442}],
       'place_id': 'ChIJL8uw1Px11kgRc7rx1AIfB6o',
       'plus_code': {'compound_code': '43X7+XV Reykjavík, Iceland',
        'global_code': '99PW43X7+XV'},
       'rating': 4.6,
       'reference': 'ChIJL8uw1Px11kgRc7rx1AIfB6o',
       'types': ['lodging', 'point_of_interest', 'establishment'],
       'user_ratings_total': 292}],
     'status': 'OK'}

``` python
location_addresses = {}
location_lat_lng = {}

# extract address and lat/long from JSON for each location
for location, info in locations_full_info.items():
    location_addresses[location] = info['results'][0]['formatted_address']
    location_lat_lng[location] = info['results'][0]['geometry']['location']

# create dataframe from dicts    
locations_df = pd.DataFrame.from_dict(location_addresses, orient='index').reset_index()
locations_df.columns = ['name', 'address']

location_lat_lng_df = pd.DataFrame.from_dict(location_lat_lng, orient='index').reset_index()
location_lat_lng_df.columns = ['name', 'lat', 'lng']

locations_merged_df = pd.merge(locations_df, location_lat_lng_df, on='name')
locations_merged_df.head(5)
```

|     | name                       | address                                 | lat       | lng        |
|-------------|---------------|---------------|-------------|-------------|
| 0   | The Reykjavik EDITION      | Austurbakki 2, 101 Reykjavík, Iceland   | 64.149924 | -21.935303 |
| 1   | Hallgrímskirkja            | Hallgrímstorg 1, 101 Reykjavík, Iceland | 64.142023 | -21.926549 |
| 2   | Sun Voyager                | Sæbraut, 101 Reykjavík, Iceland         | 64.147631 | -21.922285 |
| 3   | National Museum of Iceland | Suðurgata 41, 102 Reykjavík, Iceland    | 64.141615 | -21.948578 |
| 4   | The Settlement Exhibition  | Aðalstræti 16, 101 Reykjavík, Iceland   | 64.147387 | -21.942430 |

Since this is a small set of locations, we can pre-calculate the exact distance between each combination before running our optimization solver. The cleanest way I found to do this was to create a dataframe with one row for each combination of locations, and iterate through, calculating the walking distance for each combo. This required a bit of manual clean up afterwards, but leaves us with start (name_x) and end (name_y) locations as well as the distance between the two in feet.

It's important to mention that there are several different ways this could be made more efficient which is particularly important if this is being run on a larger data set: 1. Distances don't need to be calculated from a location to itself, this should always be 0 feet. 2. Distance from A to B should be nearly identical for B to A. I'm leaving this as is to keep the model more robust for driving directions as well. However, this would cut the number of API calls in half. 3. The distance_matrix function can take in more than two locations, however it cannot handle all twenty in this example. For simplicity, I chose to just calculate distances in pairs, though there is certainly a more performant solution for this if required. 4. Distances are static values, they only need to be calculated once for each combination of locations. These values can then be saved to disk instead of retrieved via API each time the script is run.

``` python
# create dataframe with one row per start/end location combo
location_combos_df = pd.merge(locations_df, locations_df, how='cross')

# calculate walking distance for each row
location_combos_df['distances'] = [client.distance_matrix(row[1][0], row[1][2], mode='walking', units = 'imperial') for row in location_combos_df.iterrows()]

# extract distance from JSON response
location_combos_df['distances_intermediate'] = [distance['rows'][0]['elements'][0] for distance in location_combos_df['distances']]

# extract status from JSON response
statuses = []
distances_formatted = []
for row in location_combos_df['distances_intermediate']:
    statuses.append(row['status'])
    if row['status'] == 'OK':
        distances_formatted.append(row['distance']['text'])
    else:
        # if missing, add 999,999 miles as distance
        distances_formatted.append('999999 mi')
        
# add fields to dataframe
location_combos_df['status'] = statuses
location_combos_df['distances_string'] = distances_formatted

# convert any rows in miles to feet
location_combos_df['distances_ft'] = [float(dist.split()[0]) if dist[-2:] == 'ft' else float(dist.split()[0]) * 5280 for dist in location_combos_df['distances_string']]
location_combos_df[['name_x', 'address_x', 'name_y', 'address_y', 'distances_ft']].head(5)
```

|     | name_x                | address_x                               | name_y                      | address_y                                 | dis tances_ft |
|-----------|-----------|-----------|-----------|-----------|-----------|
| 0   | The Reykjavik EDITION | Au sturbakki 2, 101 R eykjavík, Iceland | The Reykjavik EDITION       | Au sturbakki 2, 101 R eykjavík, Iceland   | 1.0           |
| 1   | The Reykjavik EDITION | Au sturbakki 2, 101 R eykjavík, Iceland | Hallgr ímskirkja            | Hall grímstorg 1, 101 R eykjavík, Iceland | 3696.0        |
| 2   | The Reykjavik EDITION | Au sturbakki 2, 101 R eykjavík, Iceland | Sun Voyager                 | Sæbraut, 101 R eykjavík, Iceland          | 3696.0        |
| 3   | The Reykjavik EDITION | Au sturbakki 2, 101 R eykjavík, Iceland | National Museum of Iceland  | Suðurgata 41, 102 R eykjavík, Iceland     | 4224.0        |
| 4   | The Reykjavik EDITION | Au sturbakki 2, 101 R eykjavík, Iceland | The S ettlement E xhibition | A ðalstræti 16, 101 R eykjavík, Iceland   | 2112.0        |


``` python
### save/load to csv to reduce api calls
# location_combos_df.to_csv('itinerary_location_combos.csv')

# location_combos_df = pd.read_csv('itinerary_location_combos.csv')
# location_combos_df['distances'] = [json.loads(row.replace("'",'"')) for row in location_combos_df['distances']]
```

### Creating the Solver and Optimizing Routes

For the optimization portion of this project, I'm using Google's [OR-Tools](https://developers.google.com/optimization) suite of optimization engines. There are two main reasons for this. First, I'm familiar with the OR-Tools workflow so I should be able to iterate quickly. Second, OR-Tools includes a routing engine specifically designed for vehicle routing which be perfect for what we're trying to accomplish here!

The routing tools require an n x n matrix of distances (n = number of locations), which we create next. We add this to a dictionary along with the length of our trip in days, and the index of the origin which is the index of our hotel. In my case the hotel is the first location in the list, so the origin is 0.

``` python
# create n x n matrix from location combinations dataframe
distance_matrix = np.array(location_combos_df['distances_ft']).reshape((len(locations),len(locations)))
```

``` python
def create_data_model(distance_matrix):
    ''' properly formats data required for solver '''
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_days"] = 4 # trip length in days
    data["origin"] = 0 # index of daily start and end point (hotel)
    return data

data = create_data_model(distance_matrix)
```

Here, I'm creating a function to parse the objective function (total feet walked during trip), and the route for each day of the trip. This will allow us to more easily utilize this data in the next step.

``` python
def return_solution(data, manager, routing, solution):
    ''' formats solution for easy manipulation
    returns: objective value and dictionary of lists for route '''
    objective_value = solution.ObjectiveValue()
    max_route_distance = 0
    plan_output = {}
    # for each day, iterate and append locations to list
    for day_index in range(data["num_days"]):
        locations = [0]
        index = routing.Start(day_index)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            # track distance for each day (not used later but good to have)
            route_distance += routing.GetArcCostForVehicle(previous_index, index, day_index)
            locations.append(manager.IndexToNode(index))
        plan_output[day_index] = locations
        max_route_distance = max(route_distance, max_route_distance)
    return objective_value, plan_output
```

There is a lot to unpack in the next code block. Feel free to read the inline comments for additional detail, but in summary the following things are being done in this function: 1. Initializing model 2. Defining distance as a measurable metric 3. Adding minimum/maximum daily distance as constraints 4. Specifying model parameters 5. Solving and returning solution

You can see the solution is output as a dictionary with keys and a list of indices for each day. These indices correspond with the order the locations were specified in the original list.

My initial reaction is to be a bit skeptical of day two, which has many more locations than the other days, but we will have to see what this looks like in reality to know for sure.

``` python
def optimize_routes(data, min_daily_ft, max_daily_ft):
    ''' initializes solver, runs solver, and outputs return_solution() '''
    # initialize model
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_days"], data["origin"])
    routing = pywrapcp.RoutingModel(manager)

    # create distance variable
    def distance_callback(from_index, to_index):
        ''' Returns the distance between the two nodes. '''
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # add distance as a constraint
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_daily_ft,  # daily maximum travel distance
        True,  # start cumul to zero
        dimension_name)

    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    for i in range(data["num_days"]):
        distance_dimension.CumulVar(routing.End(i)).RemoveInterval(0, min_daily_ft)

    # setting solver parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = 30

    # solve
    solution = routing.SolveWithParameters(search_parameters)

    # return solution
    if solution:
        return return_solution(data, manager, routing, solution)
    else:
        raise ValueError("No solution found! Confirm that all addresses are valid and min/max distances are realistic.")
        
obj_function, routes = optimize_routes(data, 10000, 80000)
routes
```

    {0: [0, 13, 10, 0],
     1: [0, 15, 11, 12, 17, 16, 19, 8, 1, 6, 2, 0],
     2: [0, 9, 7, 14, 5, 0],
     3: [0, 18, 3, 4, 0]}

This simple function just takes in our raw solution and outputs it in a more intuitive, human-readable format.

``` python
def format_routes(routes):
    ''' parses routes dictionary as a multi-line string '''
    for day, route in routes.items():
        print(f'Day {day+1}\n',' -> '.join([locations[i] for i in route]))
        
format_routes(routes)
```

    Day 1
     The Reykjavik EDITION -> DEIG workshop -> Grótta Island Lighthouse -> The Reykjavik EDITION
    Day 2
     The Reykjavik EDITION -> Bæjarins Beztu Pylsur -> The Icelandic Punk Museum -> Mat Bar -> Penninn Eymundsson Skólavörðustíg -> 12 Tónar -> Skólavörðustígur -> The Einar Jónsson Museum -> Hallgrímskirkja -> Laugavegur Reykjavík -> Sun Voyager -> The Reykjavik EDITION
    Day 3
     The Reykjavik EDITION -> Reykjavík Maritime Museum -> Aurora Reykjavík -> Anna Jóna -> Harpan Concert Hall -> The Reykjavik EDITION
    Day 4
     The Reykjavik EDITION -> Hljómskálagarðurinn -> National Museum of Iceland -> The Settlement Exhibition -> The Reykjavik EDITION

### Visualizing

The last step of this project is to visualize the routes the solver has created. In order to do this, I'm going to make use of the gmplot library, a library which greatly simplifies the process of working with Google Maps' Directions API.

This library operates using latitude and longitude so we'll need to parse these in a very specific way for each location. The following function does just that.

``` python
def parse_lat_lng(routes_dict, location_lat_lng_df):
    ''' returns lat/long coords from location_lat_lng_df for each location in routes_dict '''
    lat_lng_dict = {}
    lat_lng_strings_dict = {}

    for day, route in routes_dict.items():
        # for each day, add list of lat/long tuples to dictionary
        lat_lng_dict[day] = [(location_lat_lng_df.loc[location,['lat']][0], location_lat_lng_df.loc[location,['lng']][0]) for location in route]

    return lat_lng_dict

lat_lng_parsed = parse_lat_lng(routes, location_lat_lng_df)
```

This final function creates a Google Maps route that starts and ends at our origin point, and follows the solution provided by the solver. The map is output as an html file meaning that it is fully interactive, however, I've attached a screenshot for reference here as well.

``` python
def generate_google_map_route(locations_merged_df, lat_lng_parsed, day):
    ''' generates Google Maps page for specified day's route '''
    apikey = 'secret_key'

    # calculate average of all locations as center point of map view
    mean_lat = np.mean(locations_merged_df.loc[:,'lat'])
    mean_lng = np.mean(locations_merged_df.loc[:,'lng'])

    # define coords for origin pinpoint
    origin = tuple(locations_merged_df.loc[0,['lat', 'lng']])

    # initialize map, set camera x,y, and zoom (zoom will depend on activity distances)
    gmap = gmplot.GoogleMapPlotter(mean_lat, mean_lng, 13, apikey=apikey)

    # add location points to map, with pinpoint on origin
    gmap.directions(
        origin,
        origin,
        travel_mode = 'WALKING',
        waypoints = lat_lng_parsed[day]
    )

    # generate html page for trip on specified day
    return gmap.draw(f'trip_route_day_{day}.html')

generate_google_map_route(locations_merged_df, lat_lng_parsed, 1)
```

![png](../../../../images/Travel_Route_1.png)

### Final Thoughts

It's rewarding to see this model finding intuitive, realistic solutions. Looking at day #2 which I was initially skeptical of, the proximity of the locations to each other makes the walk totally feasible.

Hypothetically speaking, if this were an unrealistic route, I think a good approach would be to tweak the min/max daily travel distance parameters to find a more realistic solution.

There are of course a few drawbacks to this model due to its simplicity. Namely, the model does not factor in time spent at each location, and makes no attempt to chose activities based on time of day (for example, its possible two restaurants could be put back to back.) This could absolutely be solved with the addition of a couple more constraints. If I find myself coming back to this project in the future, I think this would be the best place to invest more time.

That being said, I'm looking forward to trying this out on new datasets. This should be as simple as updating the city coordinates and inputting a new set of activities. If you're interested in trying this for yourself, feel free to take a look at the Google Colab notebook here.
