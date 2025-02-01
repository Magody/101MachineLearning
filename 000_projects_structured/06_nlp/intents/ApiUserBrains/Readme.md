# Ridetel Chatbot con RNN (Seq2Seq)

#### Primero configurar el archivo "Parameters.py". Siempre que sea necesario

### Para entrenar desde 0
* python app.py train

### Para entrenar desde un checkpoint
* python app.py train [num_checkpoint]

### Para testear, con un checkpoint
* python app.py test [num_checkpoint]

### Para predecir texto directamente, con un checkpoint
* python app.py predict 1000 "hola"

### Para entrenar y testear
* python app.py all
* python app.py all [num_checkpoint]

