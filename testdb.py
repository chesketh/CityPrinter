import pathlib
import laspy

def inspect_laz(laz_path: str):
    laz_file = pathlib.Path(laz_path)
    if not laz_file.exists():
        raise FileNotFoundError(f"LAZ file not found: {laz_file}")

    with laspy.open(str(laz_file)) as laz:
        print(f"Inspecting LAZ file: {laz_file}")
        header = laz.header
        print(f"Version: {header.version}")
        print(f"Point format: {header.point_format}")
        print(f"Point count: {header.point_count}")

        # Get coordinate reference system info
        print(f"\nCoordinate System Info:")
        print(f"Raw mins (X, Y, Z): {header.mins}")
        print(f"Raw maxs (X, Y, Z): {header.maxs}")
        print(f"Scales: {header.scales}")
        print(f"Offsets: {header.offsets}")

        # Try to read a small sample of points to verify data access
        print("\nTesting point access (first 5 points):")
        las = laz.read()
        for i in range(min(5, len(las.points))):
            print(f"Point {i}: X={las.x[i]}, Y={las.y[i]}, Z={las.z[i]}")

if __name__ == "__main__":
    # Inspect the file that's causing issues
    inspect_laz("CityBuilder/lidar/seattle/King_2284_rp.laz")