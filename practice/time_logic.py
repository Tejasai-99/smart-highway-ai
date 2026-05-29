import time

# Simulated vehicle ID
vehicle_id = 1

# Vehicle enters Camera A
print(f"Vehicle {vehicle_id} entered Camera A")

# Store start time
start_time = time.time()

# Waiting time (seconds)
timeout = 10

print("Waiting for vehicle in Camera B...")

# Simulate checking
while True:

    current_time = time.time()

    elapsed_time = current_time - start_time

    # Simulate vehicle detection in Camera B
    user_input = input("Did vehicle appear in Camera B? (yes/no): ")

    if user_input.lower() == "yes":

        print("SAFE: Vehicle reached Camera B")
        break

    elif elapsed_time > timeout:

        print("ALERT: Vehicle missing!")
        break

    else:

        remaining = int(timeout - elapsed_time)

        print(f"Still waiting... {remaining} seconds left")