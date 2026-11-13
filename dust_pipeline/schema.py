"""Names and definitions shared by the DAG, the agents and the evaluation suite."""

TABLE = "saudi_dust_station_daily"
TARGET = "low_vis_next_day"

# Rows are station-days. Splits are chronological on this column whenever it is present.
TIME_COLUMN = "obs_date"

# Today's value of the thing being forecast → the "tomorrow looks like today" baseline.
PERSISTENCE_COLUMN = "low_vis_today"

# Tomorrow's raw outcome measurements. They stay in the warehouse table for analysis,
# but they are the answer, not features: a model must never train on them.
OUTCOME_COLUMNS = ["next_day_min_visibility_m", "next_day_low_vis_hours"]

# Event definition (applied per hourly report, then per day).
LOW_VIS_METERS = 5000  # visibility below 5 km
DRY_AIR_MAX_RH = 70.0  # relative humidity below 70% → dust/haze rather than fog or rain
MIN_LOW_VIS_HOURS = 2  # at least two distinct clock hours with such a report in the local day
MIN_HOURLY_OBS_PER_DAY = 12  # days with reports in fewer clock hours are too thin to label
LOCAL_UTC_OFFSET_HOURS = 3  # Asia/Riyadh, no daylight saving
