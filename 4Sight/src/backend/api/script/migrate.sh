#!/bin/bash
echo "Migrating..."
pip install Flask
pip install Flask-Migrate
flask db init
flask db migrate
flask db upgrade