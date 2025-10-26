#!/bin/bash
# /scripts/load_job.sh

# Exit immediately if a command exits with a non-zero status.
set -e

DB_HOST="${DB_HOST}"
DB_USER="${DB_USER}"
DB_NAME="${DB_NAME}"
export PGPASSWORD="${DB_PASSWORD}"

SCHEMA_FILE="/app/benchmarks/job/schema.sql"
DATA_DIR="/app/benchmarks/job/data"

echo "--- Starting JOB Benchmark Load ---"
echo "Host: ${DB_HOST}, User: ${DB_USER}, DB: ${DB_NAME}"

# 1. Create the schema
echo "Applying schema from ${SCHEMA_FILE}..."
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$SCHEMA_FILE"

# 2. Load data from CSV files
echo "Loading data from ${DATA_DIR}..."

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy aka_name FROM '${DATA_DIR}/aka_name.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy aka_title FROM '${DATA_DIR}/aka_title.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy cast_info FROM '${DATA_DIR}/cast_info.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy char_name FROM '${DATA_DIR}/char_name.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy comp_cast_type FROM '${DATA_DIR}/comp_cast_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy company_name FROM '${DATA_DIR}/company_name.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy company_type FROM '${DATA_DIR}/company_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy complete_cast FROM '${DATA_DIR}/complete_cast.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy info_type FROM '${DATA_DIR}/info_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy keyword FROM '${DATA_DIR}/keyword.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy kind_type FROM '${DATA_DIR}/kind_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy link_type FROM '${DATA_DIR}/link_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy movie_companies FROM '${DATA_DIR}/movie_companies.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy movie_info FROM '${DATA_DIR}/movie_info.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy movie_info_idx FROM '${DATA_DIR}/movie_info_idx.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy movie_keyword FROM '${DATA_DIR}/movie_keyword.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy movie_link FROM '${DATA_DIR}/movie_link.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy name FROM '${DATA_DIR}/name.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy person_info FROM '${DATA_DIR}/person_info.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy role_type FROM '${DATA_DIR}/role_type.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "\copy title FROM '${DATA_DIR}/title.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', ESCAPE '\');"

# 3. Analyze the database
echo "Running VACUUM ANALYZE..."
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;"

echo "--- JOB Benchmark Load Complete ---"