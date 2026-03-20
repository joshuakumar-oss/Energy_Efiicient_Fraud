import os
import time

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None


def main():
    if EmissionsTracker is None:
        raise ImportError("codecarbon is not installed. Install it before running carbon.py.py")

    os.makedirs("emissions_logs", exist_ok=True)

    tracker = EmissionsTracker(
        output_dir="emissions_logs",
        log_level="error",
    )

    print("Starting carbon tracking...")
    tracker.start()
    time.sleep(60)
    emissions = tracker.stop()

    print("Tracking complete!")
    print(f"Emissions: {emissions} kg CO2")
    print("\nFiles in emissions_logs:")
    print(os.listdir("emissions_logs"))


if __name__ == "__main__":
    main()
