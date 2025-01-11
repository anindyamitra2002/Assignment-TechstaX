# Assignment-Fixit

redundant or potentially unnecessary columns:

1. Location-Related Redundancies:
- `End_Lat` and `End_Lng`: These are often null and the start coordinates (`Start_Lat`, `Start_Lng`) are sufficient for most analyses, especially since we have `Distance(mi)` to capture the accident extent.
- `Number` and `Street`: These are too granular for accident prediction and are already represented by higher-level location data (City, County, State).
- `Country`: Always "US" (redundant since this is US accident data).
- `Airport_Code`: Redundant with location coordinates and only used as a reference for weather station.
- `Turning_Loop`: always false in the given data

2. Time-Related Redundancies:
- `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`: These are highly correlated with `Sunrise_Sunset` and provide similar information. `Sunrise_Sunset` is sufficient for day/night classification.
- `Weather_Timestamp`: Likely redundant with `Start_Time` unless there's a significant time difference.

3. Identification/Description:
- `ID`: Not needed for modeling (unique identifier).

4. Weather-Related Redundancies:
- `Wind_Chill(F)`: Can be derived from Temperature and Wind_Speed.
- Either `Temperature(F)` or `Wind_Chill(F)` could be redundant depending on your modeling needs.

5. Address Components:
- `Side` (Right/Left): Likely not significant for predicting accident severity.
- `Zipcode`: Redundant with City/County/State information.


redundant_columns = [
    'End_Lat',
    'End_Lng',
    'Number',
    'Street',
    'Country',
    'Airport_Code',
    'Turning_Loop',
    'Civil_Twilight',
    'Nautical_Twilight',
    'Astronomical_Twilight',
    'Weather_Timestamp',
    'ID',
    'Description',
    'Wind_Chill(F)',
    'Side',
    'Zipcode'
]


