Assignment 1 is in gentext-app, ignore main.py in this directory
generate_text prints an error message if the input is not in the model's vocabulary
For brevity, word_embedding only prints the first 10 elements

Docker commands:

docker build -t gentext-app .
docker run -p 8000:80 gentext-app

Access at: 

http://127.0.0.1:8000
http://127.0.0.1:8000/docs