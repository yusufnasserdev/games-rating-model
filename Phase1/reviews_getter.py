import csv
import os
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

df = pd.DataFrame(columns = ["ID","Reviews"])
# Read CSV file
with open('games-regression-dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        url = row[0]  # URL is in first column
        filename = 'Reviews/'+os.path.basename(url)  # Extract filename from URL
        url +=  "?see-all=reviews"
        response = requests.get(url)
        if response.status_code == 200:  # Check if request was successful
            soup = BeautifulSoup(response.text, 'html.parser')
            blocks = soup.findAll("blockquote")
            review_list = []
            for blockquote in blocks:
                review = blockquote.find('p').text
                review_list.append(review)
            if len(review_list)!=0:
                filename = re.sub(r'[^\d]+', '', filename)
                new_row = {'ID': filename,"Reviews": review_list}
                df = df._append(new_row, ignore_index=True)
            print(f'Successfully downloaded {filename}')
df.to_csv('Reviews.csv', index=False)
