import sqlite3
import datetime

# Function to create a connection to the insuarance_cop.db database
def create_connection():
    connection = sqlite3.connect('insuarnace.db')
    return connection

#ALL TABLE CREATING FUNCTIONS=================================================
# Function to create the results table if it doesn't exist
def create_company_table(connection):
    with connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS company (
            companyName TEXT NOT NULL PRIMARY KEY UNIQUE,
            compReviewScore  varchar(255) NOT NULL,
            summary TEXT NOT NULL,
            rating double NOT NULL,
            )''')

def create_product_table(connection):
    with connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS product (
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
def insert_company(connection, placeID, companyName, compReviewScore):
    connection.execute('''INSERT INTO company (placeID, companyName, compReviewScore) VALUES (?, ?, ?)''', (placeID, companyName, compReviewScore))


# Function to insert a new product into the product table
def insert_product(connection, productID, agency, agency_type, distribution_channel, product_name, claim, duration, destination, net_sale, commission, gender, age):
    connection.execute('''INSERT INTO product (productID, agency, agency_type, distribution_channel, product_name, claim, duration, destination, net_sale, commission, gender, age) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (productID, agency, agency_type, distribution_channel, product_name, claim, duration, destination, net_sale, commission, gender, age))



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

import csv

def insert_products_from_csv(conn, filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            # Convert appropriate values (e.g., int, float)
            row[0] = int(row[0])                  # id
            row[6] = int(row[6])                  # Agency
            row[8] = float(row[8])                # Agency Type
            row[9] = float(row[9])                # Distribution Channel
            row[11] = int(row[11])                # Product Name
            row[12] = int(row[12])                # Claim
            row[13] = int(row[13])                # Duration
            row[14] = float(row[14])              # Destination
            row[15] = float(row[15])              # Net Sale
            row[16] = float(row[16])              # Commission
            row[17] = int(row[17])                # Gender
            row[18] = int(row[18])                # Age
            insert_product(conn, row)

# Usage:
# insert_products_from_csv(conn, 'dataset_sample_x.csv')
