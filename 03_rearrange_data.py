import os
import sqlite3
import shutil

# Remove all files in the 'dataset' directory
dataset_dir = 'dataset'
files_to_delete = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if os.path.isfile(filename)]
for file_path in files_to_delete:
    try:
        os.unlink(file_path)
    except OSError as e:
        print(f"Error deleting {file_path}: {e}")

# Copy files from 'dataset-clusters' directory to 'dataset' directory
source_dir = 'dataset-clusters'
cluster_files = [(os.path.join(source_dir, cluster_dir, filename), os.path.join(dataset_dir, filename)) for cluster_dir in os.listdir(source_dir) for filename in os.listdir(os.path.join(source_dir, cluster_dir)) if os.path.isfile(os.path.join(source_dir, cluster_dir, filename))]
for src, dst in cluster_files:
    shutil.copy(src, dst)

# Remove the 'dataset-clusters' directory
shutil.rmtree(source_dir)

# Connect to the SQLite database
conn = sqlite3.connect('customer_faces_data.db')
c = conn.cursor()

# Retrieve all records from the 'faces' table
c.execute("SELECT id, image_path FROM customers")
rows = c.fetchall()

# Prepare a list of IDs to delete
ids_to_delete = [row[0] for row in rows if not os.path.isfile(row[1])]

# Delete records in a transaction
if ids_to_delete:
    c.executemany("DELETE FROM customers WHERE id=?", [(id,) for id in ids_to_delete])
    conn.commit()
    print(f"Deleted {len(ids_to_delete)} records because the associated pictures do not exist.")

# Close the database connection
conn.close()
