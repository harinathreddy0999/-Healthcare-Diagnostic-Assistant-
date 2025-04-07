#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
if [ "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    while ! nc -z $(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\).*/\1/p') $(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p'); do
        sleep 0.1
    done
    echo "PostgreSQL started"
fi

# Run database migrations
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Create initial superuser if needed
if [ "$CREATE_SUPERUSER" = "true" ]; then
    echo "Creating superuser..."
    python -c "
from app.db.init_db import init_db
from app.db.session import SessionLocal
init_db(SessionLocal())
"
fi

# Execute the main command
exec "$@" 