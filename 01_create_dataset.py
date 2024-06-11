import cv2
import time
import sqlite3
import os

def generate_uid():
    """Generate a unique ID based on the current Unix timestamp."""
    return int(time.time() * 1000)

def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)

def initialize_database():
    """Initialize the SQLite database and create the necessary table."""
    db_file = 'customer_faces_data.db'
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        print("Successfully connected to the database")
        c.execute('''CREATE TABLE IF NOT EXISTS customers
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_uid TEXT, customer_name TEXT, image_path TEXT)''')
        print("Table 'customers' created successfully")
        return conn, c
    except sqlite3.Error as e:
        print("SQLite error:", e)
        exit()

def capture_face_images(conn, c, customer_name, customer_uid):
    """Capture face images and save them along with metadata to the database."""
    dataset_dir = 'dataset'
    ensure_directory_exists(dataset_dir)
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    start_time = time.time()
    interval = 500
    current_time = start_time
    image_count = 0
    
    while True:
        ret, image = camera.read()
        if not ret:
            print("Failed to capture frame from the camera")
            break
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            continue
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Generating image {image_count+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if (time.time() - current_time) * 1000 >= interval and image_count < 50:
                image_name = f"{customer_uid}_{image_count+1}.jpg"
                image_path = os.path.join(dataset_dir, image_name)
                cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                print(f"Saved image {image_count+1} at {image_path}")
                current_time = time.time()
                image_count += 1
                
                try:
                    c.execute("INSERT INTO customers (customer_uid, customer_name, image_path) VALUES (?,?,?)", (customer_uid, customer_name, image_path))
                    conn.commit()
                    print(f"Image {image_count} inserted into database successfully")
                except sqlite3.Error as e:
                    print("SQLite error:", e)
                    
        cv2.imshow("Dataset Generating...", image)
        if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= 50:
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    time.sleep(1)
    
    conn, c = initialize_database()
    customer_name = input('Enter the Customer Name: ')
    customer_uid = generate_uid()
    
    print("Please get your face ready!")
    time.sleep(2)
    
    capture_face_images(conn, c, customer_name, customer_uid)
    
    conn.close()
    print("Database connection closed.")
