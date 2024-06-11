## Face Recognition with Customer Name Lookup

This project implements a face recognition system with an enhanced output: displaying the recognized customer's **full name** instead of a raw ID.

### Project Overview

* **Dynamic Customer Name Retrieval:** The system retrieves customer names directly from a database, eliminating the need for hardcoded names.
* **Data Management and Cleaning:** Scripts are provided to create, organize, and clean the dataset, ensuring its integrity for training the model.

### Setup
To install all the required modules, simply run:
```sh
pip install -r requirements.txt
```

> Obviously, it is highly advised to create a virtualenv first, but that's strictly YOUR preference

### Files (to be run in this order)

* `01_create_dataset.py`: Captures grayscale face images, saves them, and stores corresponding metadata (customer name, image path) in a SQLite database.
* `02_create_clusters.py`: Automates image clustering based on features extracted with a VGG16 model. This step helps identify potentially fake faces for manual removal, maintaining a clean dataset.
* `03_rearrange_data.py`: Manages dataset structure: clears the `dataset` directory, populates it with images from pre-clustered data, removes the temporary `dataset-clusters` folder, and ensures data integrity within the database.
* `04_train_model.py`: Trains a Local Binary Patterns Histograms (LBPH) face recognizer. It leverages a Haar Cascade Classifier for face detection, extracts face samples and labels, trains the recognizer, and saves the trained model.
* `05_make_predictions.py`: Utilizes the pre-trained LBPH model for face recognition. It captures webcam frames, detects faces, predicts identity and confidence level. If the confidence is high enough, it displays a rectangle around the face with the predicted customer's **full name** and confidence percentage.

### Assignment: Enhanced Output

This project expands upon the base functionality by retrieving and displaying the full name of the recognized customer from the database based on the predicted ID (UID).

### Exploring the SQLite Database (Optional)

Assuming you have SQLite3 installed, you can interact with the database named `customer_faces_data.db` using the following steps:

1. Open a terminal and type: `sqlite3 customer_faces_data.db`
2. You'll see a prompt: `sqlite>`.
3. Enter commands to explore the database schema and data:
    * `pragma table_info(customers);` (Shows information about the `customers` table)
    * `select * from customers;` (Retrieves all data from the `customers` table)

**Note:** These commands are for illustration purposes only. You can use various SQL queries to analyze and manage the database content.
