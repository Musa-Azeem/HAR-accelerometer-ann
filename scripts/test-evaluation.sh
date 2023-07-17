#! /bin/bash

# Run without seperating sessions
./scripts/main_nursingv1-mix-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5 --project nursingv1_projects/evaluation_tests

# Run with seperating sessions
./scripts/main_nursingv1-seperate-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5 --project nursingv1_projects/evaluation_tests
