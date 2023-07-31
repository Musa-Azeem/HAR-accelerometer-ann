#! /bin/bash

# Run with seperating sessions
./scripts/main_nursingv1-seperate-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5 --project nursingv1_projects/evaluation_tests/seperate-sessions --model resnetlstm --device 0
# ./scripts/main_nursingv1-seperate-sessions.py --n-sessions 2 --shuffle --epochs 2 --dev-size .5 --project nursingv1_projects/evaluation_tests/seperate-sessions --model resnetlstm --dataset-path data/nursingv1_100Hz_dataset --winsize 505

# Run without seperating sessions
./scripts/main_nursingv1-mix-sessions.py --n-sessions 40 --shuffle --epochs 100 --dev-size .5 --project nursingv1_projects/evaluation_tests/mixed-sessions --model resnetlstm --device 0
# ./scripts/main_nursingv1-mix-sessions.py --n-sessions 2 --shuffle --epochs 2 --dev-size .5 --project nursingv1_projects/evaluation_tests/mixed-sessions --model resnetlstm --dataset-path data/nursingv1_100Hz_dataset --winsize 505