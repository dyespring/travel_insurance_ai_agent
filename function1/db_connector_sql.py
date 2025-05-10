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
            placeID varchar(40) PRIMARY KEY UNIQUE,
            companyName TEXT NOT NULL,
            compReviewScore  varchar(255) NOT NULL,
            )''')



#ALL INSERT FUNCTIONS=================================================
# Function to insert a new company into the company table
def insert_company(connection, placeID, companyName, compReviewScore):
    connection.execute('''INSERT INTO company (placeID, companyName, compReviewScore) VALUES (?, ?, ?)''', (placeID, companyName, compReviewScore))

