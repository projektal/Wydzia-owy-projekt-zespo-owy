#Załadowanie niezbędnych modułów
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt

# Wczytanie modelu
loaded_model = load_model('model.keras')

image_dir = './test_img'
test_sample = os.listdir(image_dir)

# Pobranie listy gatunków ptaków do sprawdzania
bird_species = os.listdir('./images')
random_files = random.sample(test_sample, 10)

results = []

# Przeszukiwanie katalogu z obrazami
for filename in random_files:
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        
        image = Image.open(image_path)
        
        # Przycinanie obrazów do wspólnego rozmiaru
        image = image.resize((128, 128))
        
        # Konwwersja obrazu do tablicy numpy oraz normalizacja
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.reshape(128, 128, 3)

        # Wykonywanie predykcji
        predictions = loaded_model.predict(image.reshape(1, 128, 128, 3))
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        # Przypisanie danych do zmiennej oraz ich formatowanie
        predicted_species = bird_species[predicted_class].split(".")[1];
        predicted_species = predicted_species.replace("_"," ")
        
    
        actual_species = filename 
        match = re.search(r'\d+',actual_species)

        if match:
            start, end = match.span()
            new_text = actual_species[:start]
        else :
            new_text = actual_species

        actual_species = new_text
        actual_species = actual_species.replace("_"," ")
        actual_species = actual_species[:-1]

        results.append((image, predicted_species, actual_species, confidence))

# Wyświetlenie wyników
plt.figure(figsize=(15, 10))
correctness = 0
for i, (image, predicted_species, actual_species, confidence) in enumerate(results):

    if predicted_species == actual_species:
        result = "Prawda"
        correctness+=1
    else:
        result = "Fałsz"

    plot_text = f"Rozpoznany gatunek: {predicted_species}\nPewność: {confidence * 100:.2f}%\n Faktyczny gatunek: {actual_species}"+" \n"+result
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(plot_text, fontsize=8)
    plt.axis('off')

print("Skuteczność wynosi "+str(correctness)+" /"+str(len(random_files)))

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

