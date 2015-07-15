import csv

with open('trainLabels.csv', 'rb') as csvfile:
    filereader = csv.reader(csvfile)
    for row in filereader:
        print row