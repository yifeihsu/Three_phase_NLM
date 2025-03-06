import opendssdirect as dss

# 1. Set the data path to where your DSS files are located (optional, but useful)
# data_path = r"path/to/your/dss_files"
# dss.Basic.DataPath(data_path)

# 2. Compile your main (master) .dss file
dss.Text.Command("compile Master.dss")

# 3. Solve the power flow
dss.Solution.Solve()

# 4. Check if the solution converged
if dss.Solution.Converged():
    print("Solution Converged!")
else:
    print("Solution did not converge.")

# 5. Get desired results

## Example A: Bus Voltages

# Retrieve all bus names
all_buses = dss.Circuit.AllBusNames()

print("=== Bus Voltage Magnitudes and Angles ===")
for bus_name in all_buses:
    # Make this bus active
    dss.Circuit.SetActiveBus(bus_name)

    # Get the bus voltage magnitudes and angles
    v_mag_angle = dss.Bus.VMagAngle()

    # The data come in pairs (Vmag1, Ang1, Vmag2, Ang2, ...)
    # Format or parse as needed
    print(f"Bus: {bus_name} -> {v_mag_angle}")

## Example B: Line Currents

# Retrieve all lines
all_lines = dss.Circuit.AllElementNames()
line_elements = [line for line in all_lines if line.lower().startswith("line.")]

print("\n=== Line Currents ===")
for line_name in line_elements:
    dss.Circuit.SetActiveElement(line_name)

    # Currents returns complex values as Real1, Imag1, Real2, Imag2, ...
    # representing current in each conductor
    currents = dss.CktElement.Currents()

    # Convert to magnitude/angle, or keep real/imag parts
    print(f"Line: {line_name} -> Currents (Real/Imag): {currents}")

# 6. (Optional) Use opendssdirect commands to export or show text-based reports:
# dss.Text.Command("Show Voltages LN")
# dss.Text.Command("Show Currents")
# and so on, which produce ASCII-based reports.
