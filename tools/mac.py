import numpy as np

def MAC(chords, spans, sweeps):
    """Assumes all trapezoidal sections defined by chords and spans.

    sweeps: trailing-edge sweep angle (degrees) for each section, same
    length as spans. Used to locate the leading edge in space so the
    quarter-MAC point can be placed relative to the wing root LE.

    Returns a dict:
      mac: mean aerodynamic chord length
      mac_span_location: spanwise station (from root) where local chord
        first drops below mac
      mac_section: index of the section containing that station
      quarter_mac_le_distance: distance from the wing root leading edge
        (the absolute LE of the wing) to the quarter-chord point of the MAC
    """
    if len(chords) - 1 != len(spans) or len(spans) != len(sweeps):
        return -1

    station_points = 10000

    total_span = sum(spans)

    span_stations = np.linspace(0,total_span,station_points)
    chord_stations = np.zeros_like(span_stations)
    le_stations = np.zeros_like(span_stations)

    span_segment = total_span / (station_points - 1)

    slopes = [(chords[i+1]-chords[i])/spans[i] for i in range(len(spans))]
    le_slopes = [np.tan(np.radians(sweeps[i])) - slopes[i] for i in range(len(spans))]

    mac_sum = 0
    area_sum = 0

    span_count = 0
    segment_start = 0.0
    le_segment_start = 0.0
    traversed_span = spans[0]
    for index, y in enumerate(span_stations):
        if y > traversed_span:
            le_segment_start += le_slopes[span_count] * spans[span_count]
            segment_start = traversed_span
            span_count += 1
            traversed_span += spans[span_count]
        chord_y = slopes[span_count]*(y - segment_start) + chords[span_count]
        mac_sum += np.square(chord_y) * total_span / (station_points)
        chord_stations[index] = chord_y
        le_stations[index] = le_slopes[span_count]*(y - segment_start) + le_segment_start
        if y == 0:
            continue
        else:
            chord_prev = slopes[span_count]*(span_stations[index-1] - segment_start) + chords[span_count]
            area_sum += ( (chord_y + chord_prev) / 2 ) * span_segment

    mac = mac_sum / area_sum

    # Spanwise location of the MAC: outermost station where local chord
    # is still >= mac (chord assumed monotonically non-increasing outboard).
    mac_station_index = 0
    for index, c in enumerate(chord_stations):
        if c >= mac:
            mac_station_index = index
        else:
            break

    mac_span_location = span_stations[mac_station_index]
    cumulative_span = np.cumsum(spans)
    mac_section = int(np.searchsorted(cumulative_span, mac_span_location))

    quarter_mac_le_distance = le_stations[mac_station_index] + 0.25 * mac

    return {
        'mac': mac,
        'mac_span_location': mac_span_location,
        'mac_section': mac_section,
        'quarter_mac_le_distance': quarter_mac_le_distance,
    }

if __name__ == "__main__":
    chords = [26.25952, 4.46961]
    span = [15.00000]
    sweep = [60]
    mac_result = MAC(chords, span, sweep)
    print("Mean Aerodynamic Chord Results:")
    print(f"MAC length in model's length unit: {mac_result['mac']}.")
    print(f"Located at span location: {mac_result['mac_span_location']}.")
    print(f"With distance from front of wing: {mac_result['quarter_mac_le_distance']}.")
    print(f"Located in wing section: {mac_result['mac_section']}.")