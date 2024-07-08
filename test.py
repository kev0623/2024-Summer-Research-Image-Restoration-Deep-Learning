import os

directory = 'C://Users//link4//Documents//2024_Summer_Research//Main_Project'

# Check write permission
if os.access(directory, os.W_OK):
    print(f'You have write permissions for {directory}')
else:
    print(f'You do NOT have write permissions for {directory}')
