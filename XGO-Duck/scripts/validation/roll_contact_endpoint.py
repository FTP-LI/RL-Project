"""Read current post-forward contact fields without changing the sensor cache."""
def found_after_forward(sensor, width):
    # Reward computation may already have cached the pre-integration contact
    # sample. Read the live sensordata view without invalidating that cache or
    # updating counters, history or any policy observation.
    data = sensor._extract_sensor_data().found
    assert data is not None and data.ndim == 2 and data.shape[1] % width == 0
    return (data > 0).reshape(data.shape[0], width, -1).any(dim=-1)
