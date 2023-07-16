#! /bin/bash

# Run without seperating sessions
./scripts/main_nursingv1-mix-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5

# Run with seperating sessions
./scripts/main_nursingv1-seperate-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5
