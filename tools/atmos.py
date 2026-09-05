import numpy as np
from ambiance import Atmosphere

def get_atmosphere(altitude):
    """Get the density at a specific altitude, specified in ft."""
    altitude_meters = altitude * 0.3048
    density = Atmosphere(altitude_meters).density
    temperature = Atmosphere(altitude_meters).temperature
    density_imperial = density * 0.0685218 / 35.3147 # Convert kg/m3 to slugs/ft3
    temperature_imperial = temperature * 1.8
    return density_imperial, temperature_imperial

if __name__ == "__main__":
    altitudes = np.linspace(0,10000,11) # ft
    GAMMA = 1.4
    R = 1716 # ft*lbf/slug/R or s2/ft2/R
    for altitude in altitudes:
        rho,T = get_atmosphere(altitude)

        print("-"*60)
        print(f"Altitude = {altitude} ft")
        print(f"Density = {rho} slugs/ft^3")
        print(f"Temperature = {T} °R")
        print(f"Speed of sound: {np.sqrt(R * GAMMA * T)}")
        print("-"*60)