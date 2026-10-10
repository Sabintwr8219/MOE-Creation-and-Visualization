# Agreed production CSV schemas

All four outputs contain **observed segment-time cells only** (`waypoint_count > 0`). The seven leading columns are identical, in order:

`link_key,segment_section_ft,segment_length_ft,start_time,end_time,waypoint_count,unique_veh_count`

| CSV | Remaining columns (order preserved) |
|---|---|
| `speed_reduction.csv` | `avg_speed_mph,posted_speed_limit_mph,speed_reduction_pct` |
| `slow_movement.csv` | `slow_movement_count,slow_movement_pct` |
| `dsh.csv` | `dsh_trip_count,DSH` |
| `travel_time.csv` | `avg_speed_mph,speed_estimated_travel_time_s` |

- `segment_section_ft` is a link-local range such as `0-500` or `500-602.8663`.
- `unique_veh_count` is the count of **distinct journeys per link-section-time cell**, not necessarily globally distinct physical vehicles. Re-entry as a new journey counts again.
- Speed reduction is clamped at 0 for cases where a valid mean speed exceeds the posted limit. Missing/nonpositive posted limit gives a blank reduction.
- `avg_speed_mph` appears in speed reduction and travel-time CSVs; no separate average-speed CSV.
- Travel time is a **speed-based estimate** derived from mean speed and actual section length. No interpolation or traversal-count fields in CSV.
- Slow movement and DSH are copied from saved Parquet as provisional metrics.
- Empty grid cells are omitted from final CSVs; `waypoint_count=0` and `unique_veh_count=0` conceptually, but their other MOEs are missing, not numeric zero.
