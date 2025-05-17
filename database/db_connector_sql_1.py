import sqlite3
import datetime
import csv

# Function to create a connection to the insuarance_cop.db database
def create_connection():
    connection = sqlite3.connect('insuarnace.db')
    return connection

#ALL TABLE CREATING FUNCTIONS=================================================
# Function to create the results table if it doesn't exist
def create_company_table(connection):
    with connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS company (
                placeID TEXT NOT NULL,
                companyName TEXT NOT NULL PRIMARY KEY UNIQUE,
                compReviewScore TEXT NOT NULL,
                summary TEXT NOT NULL,
                rating REAL NOT NULL
            )
        ''')

def create_product_table(connection):
    with connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS product (
                productID TEXT NOT NULL PRIMARY KEY UNIQUE,
                agency TEXT,
                agency_type TEXT,
                distribution_channel TEXT,
                product_name TEXT,
                claim TEXT,
                duration INTEGER,
                destination TEXT,
                net_sale REAL,
                commission REAL,
                gender TEXT,
                age INTEGER
            )
        ''')


#ALL INSERT FUNCTIONS=================================================
# Function to insert a new company into the company table
def insert_company(connection, placeID, companyName, compReviewScore, summary, rating):
    connection.execute('''
        INSERT INTO company (placeID, companyName, compReviewScore, summary, rating)
        VALUES (?, ?, ?, ?, ?)
    ''', (placeID, companyName, compReviewScore, summary, rating))
    connection.commit()



# Function to insert a new product into the product table
def insert_product(connection, productID, agency, agency_type, distribution_channel,
                   product_name, claim, duration, destination,
                   net_sale, commission, gender, age):
    connection.execute('''
        INSERT INTO product (productID, agency, agency_type, distribution_channel,
                             product_name, claim, duration, destination,
                             net_sale, commission, gender, age)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (productID, agency, agency_type, distribution_channel, product_name,
          claim, duration, destination, net_sale, commission, gender, age))
    connection.commit()


# insert rating sample 

# insert_company(conn,
#                    companyName="AIA Insurance",
#                    compReviewScore="4.2/5",
#                    summary="Trusted insurer for global travel",
#                    rating=4.2)



# For product table insert sample function

# way to insert a Row into the database
# sample_row = (
#     0, "CBH", "Travel A", "Offline", "Comprehensive Plan", "No",
#     186, "MALAYSIA", -29.0, 9.57, "F", 81
# )

# conn = create_connection()
# create_product_table(conn)
# insert_product(conn, sample_row)
# print("Sample product inserted successfully.")

# Bulk Insert From CSV

def insert_products_from_csv(conn, filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            # Convert data types accordingly
            productID = row[0]
            agency = row[1]
            agency_type = row[2]
            distribution_channel = row[3]
            product_name = row[4]
            claim = row[5]
            duration = int(row[6])
            destination = row[7]
            net_sale = float(row[8])
            commission = float(row[9])
            gender = row[10]
            age = int(row[11])
            
            insert_product(conn, productID, agency, agency_type, distribution_channel,
                           product_name, claim, duration, destination,
                           net_sale, commission, gender, age)


# ======= USAGE test =======
if __name__ == "__main__":
    conn = create_connection()
    create_company_table(conn)
    create_product_table(conn)

    # Sample company insert
    insert_company(conn, "AIA001", "AIA Insurance", "4.2/5", "Trusted insurer for global travel", 4.2)

    # Sample bulk product insert
    # insert_products_from_csv(conn, 'dataset_sample_x.csv')
    conn.close()
