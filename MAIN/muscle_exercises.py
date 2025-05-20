# muscle_exercises.py
def get_exercise_for_muscle(muscle_name):
    # Define a dictionary with muscle names as keys and exercises as values
    exercises = {
        "Left Shoulder (Deltoid)": [
            "Shoulder Press",
            "Lateral Raises",
            "Front Raises",
            "Reverse Fly",
            "Upright Row"
        ],
        "Right Shoulder (Deltoid)": [
            "Shoulder Press",
            "Lateral Raises",
            "Front Raises",
            "Reverse Fly",
            "Upright Row"
        ],
        "Left Elbow (Triceps)": [
            "Tricep Dips",
            "Tricep Kickbacks",
            "Skull Crushers",
            "Overhead Tricep Extension"
        ],
        "Right Elbow (Triceps)": [
            "Tricep Dips",
            "Tricep Kickbacks",
            "Skull Crushers",
            "Overhead Tricep Extension"
        ],
        "Right Trapezius":[
            "Farmers carry",
            "Dumbell lateral rises",
        ],
        "Left Trapezius": [
            "Farmers carry",
            "Dumbell lateral rises",
        ],
        "Chest (Pectorals)": [
            "Push-ups",
            "Bench Press",
            "Chest Fly",
            "Dumbbell Press"
        ],
        "Left Bicep": [
            "Bicep Curls",
            "Hammer Curls",
            "Chin-ups"
        ],
        "Right Bicep": [
            "Bicep Curls",
            "Hammer Curls",
            "Chin-ups"
        ]
    }

    # Return the exercise for the given muscle name as a list, or a default list if not found
    return exercises.get(muscle_name, ["Exercise not found"])
