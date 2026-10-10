"""Partition matched CV rows by vehicle without ever reading CV GPS columns."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths
from progress import Progress

CV_COLUMNS = ["journey_id", "capture_time", "local_time", "link_key", "speed_mph", "x_ft", "y_ft"]
CV_SCHEMA = pa.schema([
    ("journey_id", pa.string()), ("capture_time", pa.float64()),
    ("local_time", pa.timestamp("ns")), ("link_key", pa.string()),
    ("speed_mph", pa.float64()), ("x_ft", pa.float64()), ("y_ft", pa.float64()),
])


# Keep worker count bounded by partitions and the measured memory budget.
def resolve_workers(requested=None):
    workers = paths.PARALLEL_WORKERS if requested is None else int(requested)
    if workers < 1 or workers > paths.JOURNEY_PARTITIONS:
        raise ValueError(f"Choose 1 to {paths.JOURNEY_PARTITIONS} workers")
    return workers


# Clear only generated temporary folders beneath this offset output directory.
def reset_generated_directory(directory):
    directory = Path(directory).resolve()
    if paths.OUTPUT_DIR.resolve() not in directory.parents:
        raise RuntimeError(f"Refusing to remove folder outside offset output: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


# Record source identity so another hour or changed CSV cannot reuse stale partitions.
def signature(max_cv_rows):
    stat = paths.CV_FILE.stat()
    return dict(source=str(paths.CV_FILE.resolve()), size=stat.st_size,
                modified_ns=stat.st_mtime_ns, schema="offset_only_v1",
                columns=CV_COLUMNS, partitions=paths.JOURNEY_PARTITIONS,
                max_cv_rows=max_cv_rows)


# Partition every vehicle into one file while preserving within-vehicle row order.
def build_journey_partitions(rebuild=False, max_cv_rows=None):
    directory = (paths.JOURNEY_PARTITION_DIR if max_cv_rows is None else
                 paths.OUTPUT_DIR / f"CV_Journey_Partitions_sample_{max_cv_rows}")
    manifest_file = directory / "manifest.json"
    if not rebuild and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature(max_cv_rows) and all(
            (directory / f"journeys_{i:03d}.parquet").is_file()
            for i in range(paths.JOURNEY_PARTITIONS)
        ):
            print(f"Using offset-only partitions: {manifest['rows_written']:,} rows")
            return sorted(directory.glob("journeys_*.parquet")), manifest["rows_written"]

    with paths.CV_FILE.open("r", encoding="utf-8-sig", newline="") as stream:
        header = set(next(csv.reader(stream)))
    vehicle = "journey_id" if "journey_id" in header else "veh_id"
    required = set(CV_COLUMNS) - {"journey_id"}
    if vehicle not in header or required - header:
        raise ValueError(f"Matched CV needs {vehicle} and {sorted(required)}")
    temporary = directory.with_name(directory.name + "_building")
    reset_generated_directory(temporary)
    writers = [None] * paths.JOURNEY_PARTITIONS
    files = [temporary / f"journeys_{i:03d}.parquet" for i in range(paths.JOURNEY_PARTITIONS)]
    read, written = 0, 0
    progress = Progress("Offset-only CV partitioning", paths.REPORT_INTERVAL_SECONDS)
    try:
        reader = pd.read_csv(paths.CV_FILE,
                             usecols=sorted(required | {vehicle}),
                             dtype={vehicle: "string", "link_key": "string"},
                             chunksize=paths.CSV_CHUNK_SIZE)
        for chunk in reader:
            if max_cv_rows is not None:
                chunk = chunk.iloc[:max_cv_rows - read]
            if chunk.empty:
                break
            read += len(chunk)
            if vehicle != "journey_id":
                chunk = chunk.rename(columns={vehicle: "journey_id"})
            chunk = chunk[chunk["journey_id"].notna()].copy()
            chunk["journey_id"] = chunk["journey_id"].astype("string")
            chunk["link_key"] = chunk["link_key"].astype("string")
            for name in ("capture_time", "speed_mph", "x_ft", "y_ft"):
                chunk[name] = pd.to_numeric(chunk[name], errors="coerce")
            chunk["local_time"] = pd.to_datetime(chunk["local_time"], errors="coerce")
            bucket = (pd.util.hash_pandas_object(chunk["journey_id"], index=False)
                      .to_numpy(dtype=np.uint64) % np.uint64(paths.JOURNEY_PARTITIONS))
            for number in np.unique(bucket):
                number = int(number)
                table = pa.Table.from_pandas(chunk.loc[bucket == number, CV_COLUMNS],
                                             schema=CV_SCHEMA, preserve_index=False, safe=False)
                if writers[number] is None:
                    writers[number] = pq.ParquetWriter(files[number], CV_SCHEMA,
                                                        compression=paths.PARQUET_COMPRESSION)
                writers[number].write_table(table)
                written += table.num_rows
            progress.report(read)
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
    empty = pa.Table.from_pylist([], schema=CV_SCHEMA)
    for file in files:
        if not file.is_file():
            pq.write_table(empty, file, compression=paths.PARQUET_COMPRESSION)
    (temporary / "manifest.json").write_text(
        json.dumps(dict(signature=signature(max_cv_rows), rows_read=read,
                        rows_written=written), indent=2), encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)
    progress.report(read, force=True)
    print(f"Offset-only partitions complete: {written:,} rows in {len(files)} files")
    return sorted(directory.glob("journeys_*.parquet")), written
