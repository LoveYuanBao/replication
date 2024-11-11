class GeoGridPrecisionConverter:
    def __init__(self, top_left, bottom_right, precision):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.precision = precision
        self.lat_step = precision
        self.lon_step = precision
        self.num_cols = int((bottom_right[1] - top_left[1]) / self.lon_step)
        self.num_rows = int((top_left[0] - bottom_right[0]) / self.lat_step)

    def latlon_to_grid(self, lat, lon):
        """Convert latitude and longitude to grid coordinates based on precision."""
        grid_x = int((lon - self.top_left[1]) / self.lon_step)
        grid_y = int((self.top_left[0] - lat) / self.lat_step)
        return grid_x, grid_y

    def grid_to_latlon(self, grid_x, grid_y):
        """Convert grid coordinates back to latitude and longitude based on precision."""
        lat = self.top_left[0] - (grid_y * self.lat_step)
        lon = self.top_left[1] + (grid_x * self.lon_step)
        return lat, lon